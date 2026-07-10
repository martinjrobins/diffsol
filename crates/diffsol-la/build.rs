#[cfg(feature = "cuda")]
fn cuda_main() -> Result<(), String> {
    // Tell cargo to invalidate the built crate whenever files of interest changes.

    use std::{env, path::PathBuf, process::Command};
    let kernel_dir = "src/cuda_kernels";
    println!("cargo:rerun-if-changed={}", kernel_dir);
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Specify the desired architecture version.
    let arch = "compute_86"; // For example, using SM 8.6 (Ampere architecture).
    let code = "sm_86"; // For the same SM 8.6 (Ampere architecture).

    // build the cuda kernels
    let ptx_file = out_dir.join("diffsol.ptx");
    let input_file = PathBuf::from(kernel_dir).join("all.cu");

    let mut nvcc_call = Command::new("nvcc");
    nvcc_call
        .arg("-ptx")
        .arg(&input_file)
        .arg("-o")
        .arg(&ptx_file)
        .arg(format!("-arch={}", arch))
        .arg(format!("-code={}", code));
    let nvcc_status = nvcc_call.status().unwrap();

    assert!(
        nvcc_status.success(),
        "Failed to compile CUDA source to PTX."
    );
    Ok(())
}

fn main() -> Result<(), String> {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "cuda")]
    cuda_main()?;

    Ok(())
}
