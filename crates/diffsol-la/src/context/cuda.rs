use std::{
    collections::HashMap,
    env, fs,
    sync::{Arc, LazyLock, Mutex},
};

use cudarc::{
    driver::{CudaContext as CudaDevice, CudaFunction, CudaModule, CudaStream},
    nvrtc::Ptx,
};

use crate::{cuda_error, error::LaError, ScalarCuda};

static DEVICES: LazyLock<Mutex<CudaGlobalContext>> =
    LazyLock::new(|| Mutex::new(CudaGlobalContext::new()));

struct CudaGlobalContext {
    devices: HashMap<usize, (Arc<CudaDevice>, Arc<CudaModule>)>,
}

impl CudaGlobalContext {
    fn new() -> Self {
        let devices = HashMap::new();
        Self { devices }
    }
    fn get(&self, ordinal: usize) -> Option<&(Arc<CudaDevice>, Arc<CudaModule>)> {
        self.devices.get(&ordinal)
    }
    fn insert(&mut self, ordinal: usize, device: Arc<CudaDevice>, module: Arc<CudaModule>) {
        self.devices.insert(ordinal, (device, module));
    }
}

/// Context for the CUDA backend.
///
/// Carries a CUDA stream and the batch count `nbatch` which determines how
/// many independent ODE systems are solved simultaneously via 2D grid kernel
/// launches.  All vectors and matrices created with this context share the
/// same batch dimension.
#[derive(Clone, Debug)]
pub struct CudaContext {
    pub(crate) stream: Arc<CudaStream>,
    nbatch: usize,
}

impl CudaContext {
    /// Compiles the PTX files for the given device.
    fn compile_ptx(device: &Arc<CudaDevice>) -> Result<Arc<CudaModule>, LaError> {
        let out_dir = env!("OUT_DIR");
        // module in diffsol.ptx
        let ptx_file = format!("{}/diffsol.ptx", out_dir);
        // check if the file exists
        if fs::metadata(&ptx_file).is_err() {
            return Err(cuda_error!(
                Other,
                format!("PTX file not found: {}", ptx_file)
            ));
        }
        // compile the PTX file
        let ptx = Ptx::from_file(&ptx_file);
        let module = device.load_module(ptx).map_err(|e| {
            cuda_error!(
                Other,
                format!("Failed to load module from PTX file {}: {}", ptx_file, e)
            )
        })?;
        Ok(module)
    }

    /// Gets the device for the given ordinal. If the device is not already created, it creates a new one.
    pub fn get_device_and_module(
        ordinal: usize,
    ) -> Result<(Arc<CudaDevice>, Arc<CudaModule>), LaError> {
        let mut devices = DEVICES.lock().unwrap();
        let (device, module) = match devices.get(ordinal) {
            Some(dev_mod) => dev_mod.clone(),
            None => {
                let device = CudaDevice::new(ordinal)
                    .map_err(|e| cuda_error!(CudaInitializationError, e.to_string()))?;
                let module = Self::compile_ptx(&device)?;
                devices.insert(ordinal, device.clone(), module.clone());
                (device, module)
            }
        };
        device.bind_to_thread().unwrap();
        Ok((device, module))
    }

    /// Creates a new CudaContext with the given ordinal and creates a new non-default stream.
    pub fn new_with_stream(ordinal: usize) -> Result<Self, LaError> {
        let (device, _module) = Self::get_device_and_module(ordinal)?;
        let stream = device
            .new_stream()
            .map_err(|e| cuda_error!(Other, format!("Failed to create new stream: {}", e)))?;
        Ok(Self { stream, nbatch: 1 })
    }

    /// Creates a new CudaContext with the given ordinal (uses default stream).
    pub fn new(ordinal: usize) -> Result<Self, LaError> {
        let (device, _module) = Self::get_device_and_module(ordinal)?;
        let stream = device.default_stream();
        Ok(Self { stream, nbatch: 1 })
    }

    pub fn with_nbatch(mut self, nbatch: usize) -> Self {
        assert!(nbatch > 0, "nbatch must be > 0");
        self.nbatch = nbatch;
        self
    }

    pub(crate) fn function<T: ScalarCuda>(&self, kernel_name: &str) -> CudaFunction {
        let ordinal = self.stream.context().ordinal();
        let (_device, module) =
            Self::get_device_and_module(ordinal).expect("Failed to get device and module");
        let kernel_name = format!("{}_{}", kernel_name, T::as_str());
        module
            .load_function(kernel_name.as_str())
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to load function {} from module diffsol. Error: {}",
                    kernel_name, e
                )
            })
    }
}

impl Default for CudaContext {
    fn default() -> Self {
        Self::new(0).unwrap()
    }
}

impl crate::Context for CudaContext {
    fn nbatch(&self) -> usize {
        self.nbatch
    }
    fn clone_with_nbatch(&self, nbatch: usize) -> Result<Self, LaError> {
        assert!(nbatch > 0, "nbatch must be > 0");
        Ok(Self {
            stream: self.stream.clone(),
            nbatch,
        })
    }
}
