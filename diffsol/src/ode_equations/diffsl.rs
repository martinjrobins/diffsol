use core::panic;
#[cfg(feature = "diffsl-external-dynamic")]
use diffsl::ExternalDynModule;
use num_traits::{One, Zero};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::any::TypeId;
use std::cell::RefCell;
use std::ops::MulAssign;
#[cfg(feature = "diffsl-external-dynamic")]
use std::path::PathBuf;

#[cfg(feature = "diffsl-external")]
use diffsl::execution::external::{ExternSymbols, ExternalModule};
use diffsl::{
    discretise::DiscreteModel,
    execution::{
        module::{
            CodegenModule, CodegenModuleCompile, CodegenModuleEmit, CodegenModuleJit,
            CodegenModuleLink,
        },
        scalar::Scalar as DiffSlScalar,
    },
    parser::parse_ds_string,
    Compiler, ObjectModule,
};

use crate::{
    error::DiffsolError, jacobian::JacobianColoring, matrix::sparsity::MatrixSparsity,
    op::nonlinear_op::NonLinearOpJacobian, ConstantOp, ConstantOpSens, ConstantOpSensAdjoint,
    LinearOp, LinearOpTranspose, Matrix, MatrixHost, NonLinearOp, NonLinearOpAdjoint,
    NonLinearOpSens, NonLinearOpSensAdjoint, OdeEquations, OdeEquationsRef, Op, Scale, Vector,
    VectorHost,
};

/// Context for the ODE equations specified using the [DiffSL language](https://martinjrobins.github.io/diffsl/).
///
/// This contains the compiled code and the data structures needed to evaluate the ODE equations.
pub struct DiffSlContext<M: Matrix<T: DiffSlScalar>, CG: CodegenModule> {
    compiler: Compiler<CG, M::T>,
    data: RefCell<Vec<M::T>>,
    ddata: RefCell<Vec<M::T>>,
    sens_data: RefCell<Vec<M::T>>,
    tmp: RefCell<M::V>,
    tmp2: RefCell<M::V>,
    tmp_root: RefCell<M::V>,
    tmp2_root: RefCell<M::V>,
    tmp_out: RefCell<M::V>,
    tmp2_out: RefCell<M::V>,
    nstates: usize,
    nroots: usize,
    nparams: usize,
    model_index: u32,
    has_mass: bool,
    has_root: bool,
    has_reset: bool,
    has_out: bool,
    nout: usize,
    ctx: M::C,
    rhs_state_deps: Vec<(usize, usize)>,
    rhs_input_deps: Vec<(usize, usize)>,
    mass_state_deps: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiffSlExternalObject {
    scalar_type: DiffSlExternalScalarType,
    object: Vec<u8>,
    rhs_state_deps: Vec<(usize, usize)>,
    rhs_input_deps: Vec<(usize, usize)>,
    mass_state_deps: Vec<(usize, usize)>,
    include_sensitivities: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DiffSlExternalScalarType {
    F32,
    F64,
}

fn diffsl_external_scalar_type<T: DiffSlScalar>() -> Result<DiffSlExternalScalarType, DiffsolError>
{
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        Ok(DiffSlExternalScalarType::F32)
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        Ok(DiffSlExternalScalarType::F64)
    } else {
        Err(DiffsolError::Other(format!(
            "DiffSl external object does not support scalar type {}",
            std::any::type_name::<T>()
        )))
    }
}

impl<M: Matrix<T: DiffSlScalar>, CG: CodegenModule> DiffSlContext<M, CG> {
    fn new_common(
        compiler: Compiler<CG, M::T>,
        rhs_state_deps: Vec<(usize, usize)>,
        rhs_input_deps: Vec<(usize, usize)>,
        mass_state_deps: Vec<(usize, usize)>,
        ctx: M::C,
    ) -> Result<Self, DiffsolError> {
        let (nstates, nparams, nout, _ndata, nroots, has_mass, has_reset) = compiler.get_dims();
        let has_root = nroots > 0;
        let has_out = nout > 0;
        let data = RefCell::new(compiler.get_new_data());
        let ddata = RefCell::new(compiler.get_new_data());
        let sens_data = RefCell::new(compiler.get_new_data());
        let tmp = RefCell::new(M::V::zeros(nstates, ctx.clone()));
        let tmp2 = RefCell::new(M::V::zeros(nstates, ctx.clone()));
        let tmp_root = RefCell::new(M::V::zeros(nroots, ctx.clone()));
        let tmp2_root = RefCell::new(M::V::zeros(nroots, ctx.clone()));
        let tmp_out = RefCell::new(M::V::zeros(nout, ctx.clone()));
        let tmp2_out = RefCell::new(M::V::zeros(nout, ctx.clone()));
        let model_index = 0;

        Ok(Self {
            compiler,
            data,
            ddata,
            sens_data,
            nparams,
            nstates,
            tmp,
            tmp2,
            tmp_root,
            tmp2_root,
            tmp_out,
            tmp2_out,
            nroots,
            nout,
            has_mass,
            has_root,
            has_reset,
            has_out,
            ctx,
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            model_index,
        })
    }
}

#[cfg(feature = "diffsl-external-dynamic")]
impl<M: Matrix<T: DiffSlScalar>> DiffSlContext<M, ExternalDynModule<M::T>> {
    pub fn new_external_dynamic(
        path: impl Into<PathBuf>,
        nthreads: usize,
        rhs_state_deps: Vec<(usize, usize)>,
        rhs_input_deps: Vec<(usize, usize)>,
        mass_state_deps: Vec<(usize, usize)>,
        ctx: M::C,
    ) -> Result<Self, DiffsolError> {
        let mode = match nthreads {
            0 => diffsl::execution::compiler::CompilerMode::MultiThreaded(None),
            1 => diffsl::execution::compiler::CompilerMode::SingleThreaded,
            _ => diffsl::execution::compiler::CompilerMode::MultiThreaded(Some(nthreads)),
        };
        let module = ExternalDynModule::new(path)
            .map_err(|e| DiffsolError::DiffslCompilerError(e.to_string()))?;
        let compiler = Compiler::from_codegen_module(module, mode)
            .map_err(|e| DiffsolError::DiffslCompilerError(e.to_string()))?;

        Self::new_common(
            compiler,
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            ctx,
        )
    }
}

#[cfg(feature = "diffsl-external")]
impl<M: Matrix<T: DiffSlScalar + ExternSymbols>> DiffSlContext<M, ExternalModule<M::T>> {
    pub fn new_external(
        nthreads: usize,
        rhs_state_deps: Vec<(usize, usize)>,
        rhs_input_deps: Vec<(usize, usize)>,
        mass_state_deps: Vec<(usize, usize)>,
        ctx: M::C,
    ) -> Result<Self, DiffsolError> {
        let mode = match nthreads {
            0 => diffsl::execution::compiler::CompilerMode::MultiThreaded(None),
            1 => diffsl::execution::compiler::CompilerMode::SingleThreaded,
            _ => diffsl::execution::compiler::CompilerMode::MultiThreaded(Some(nthreads)),
        };
        let module = ExternalModule::default();
        let compiler = Compiler::from_codegen_module(module, mode)
            .map_err(|e| DiffsolError::DiffslCompilerError(e.to_string()))?;

        Self::new_common(
            compiler,
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            ctx,
        )
    }
}

impl<M: Matrix<T: DiffSlScalar>, CG: CodegenModuleLink + CodegenModuleJit> DiffSlContext<M, CG> {
    fn new_from_object(
        object: Vec<u8>,
        nthreads: usize,
        rhs_state_deps: Vec<(usize, usize)>,
        rhs_input_deps: Vec<(usize, usize)>,
        mass_state_deps: Vec<(usize, usize)>,
        ctx: M::C,
    ) -> Result<Self, DiffsolError> {
        let mode = match nthreads {
            0 => diffsl::execution::compiler::CompilerMode::MultiThreaded(None),
            1 => diffsl::execution::compiler::CompilerMode::SingleThreaded,
            _ => diffsl::execution::compiler::CompilerMode::MultiThreaded(Some(nthreads)),
        };
        let compiler = Compiler::from_object_file(object.clone(), mode)
            .map_err(|e| DiffsolError::DiffslCompilerError(e.to_string()))?;

        Self::new_common(
            compiler,
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            ctx,
        )
    }
}

impl<M: Matrix<T: DiffSlScalar>, CG: CodegenModuleCompile + CodegenModuleJit> DiffSlContext<M, CG> {
    /// Create a new context for the ODE equations specified using the [DiffSL language](https://martinjrobins.github.io/diffsl/).
    /// The input parameters are not initialized and must be set using the [OdeEquations::set_params] function before solving the ODE.
    ///
    /// # Arguments
    ///
    /// * `text` - The text of the ODE equations in the DiffSL language.
    /// * `nthreads` - The number of threads to use for code generation (0 for automatic, 1 for single-threaded).
    ///
    pub fn new(text: &str, nthreads: usize, ctx: M::C) -> Result<Self, DiffsolError> {
        let mode = match nthreads {
            0 => diffsl::execution::compiler::CompilerMode::MultiThreaded(None),
            1 => diffsl::execution::compiler::CompilerMode::SingleThreaded,
            _ => diffsl::execution::compiler::CompilerMode::MultiThreaded(Some(nthreads)),
        };
        let options = diffsl::execution::compiler::CompilerOptions {
            mode,
            ..Default::default()
        };
        let model =
            parse_ds_string(text).map_err(|e| DiffsolError::DiffslParserError(e.to_string()))?;
        let mut model = DiscreteModel::build("diffsol", &model)
            .map_err(|e| DiffsolError::DiffslCompilerError(e.as_error_message(text)))?;
        let compiler = Compiler::from_discrete_model(&model, options, Some(text))
            .map_err(|e| DiffsolError::DiffslCompilerError(e.to_string()))?;
        let rhs_state_deps = model.take_rhs_state_deps();
        let rhs_input_deps = model.take_rhs_input_deps();
        let mass_state_deps = model.take_mass_state_deps();

        Self::new_common(
            compiler,
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            ctx,
        )
    }
}

impl<M: Matrix<T: DiffSlScalar>, CG: CodegenModuleJit + CodegenModuleCompile> Default
    for DiffSlContext<M, CG>
{
    fn default() -> Self {
        Self::new(
            "
            u { y = 1 }
            F { -y }
            out { y }
        ",
            1,
            M::C::default(),
        )
        .unwrap()
    }
}

/// DiffSl implementation of ODE equations. This uses the [DiffSL language](https://martinjrobins.github.io/diffsl/) to specify the ODE equations.
///
/// The DiffSL code is compiled into the [DiffSlContext] which is used to evaluate the ODE equations. After compilation,
/// if the matrix type is sparse, the sparsity patterns of the Jacobians are extracted from the compiled code for use in the ODE solver.
pub struct DiffSl<M: Matrix<T: DiffSlScalar>, CG: CodegenModule> {
    context: DiffSlContext<M, CG>,
    include_sensitivities: bool,
    mass_sparsity: Option<M::Sparsity>,
    mass_coloring: Option<JacobianColoring<M>>,
    mass_transpose_sparsity: Option<M::Sparsity>,
    mass_transpose_coloring: Option<JacobianColoring<M>>,
    rhs_sparsity: Option<M::Sparsity>,
    rhs_coloring: Option<JacobianColoring<M>>,
    rhs_adjoint_sparsity: Option<M::Sparsity>,
    rhs_adjoint_coloring: Option<JacobianColoring<M>>,
    rhs_sens_sparsity: Option<M::Sparsity>,
    rhs_sens_coloring: Option<JacobianColoring<M>>,
    rhs_sens_adjoint_sparsity: Option<M::Sparsity>,
    rhs_sens_adjoint_coloring: Option<JacobianColoring<M>>,
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> DiffSl<M, CG> {
    /// Create a `DiffSl` instance from a pre-compiled `DiffSlContext`.
    ///
    /// This function extracts the sparsity patterns and Jacobian colorings from the compiled
    /// context if the matrix type is sparse. The sparsity patterns are used by ODE solvers
    /// to efficiently compute Jacobians using finite differences with coloring.
    ///
    /// # Arguments
    ///
    /// * `context` - A pre-compiled DiffSL context containing the compiled code
    /// * `include_sensitivities` - Whether to extract sparsity patterns for sensitivity computations.
    ///   If `true`, extracts sparsity patterns for forward and adjoint sensitivities. Set to `true`
    ///   if you plan to compute sensitivities or adjoints.
    ///
    /// # Returns
    ///
    /// A new `DiffSl` instance with sparsity patterns extracted (if applicable).
    ///
    /// # Note
    ///
    /// For dense matrices, this function simply wraps the context without extracting sparsity patterns.
    pub fn from_context(context: DiffSlContext<M, CG>, include_sensitivities: bool) -> Self {
        let mut ret = Self {
            context,
            include_sensitivities,
            mass_coloring: None,
            mass_sparsity: None,
            mass_transpose_coloring: None,
            mass_transpose_sparsity: None,
            rhs_coloring: None,
            rhs_sparsity: None,
            rhs_adjoint_coloring: None,
            rhs_adjoint_sparsity: None,
            rhs_sens_coloring: None,
            rhs_sens_sparsity: None,
            rhs_sens_adjoint_coloring: None,
            rhs_sens_adjoint_sparsity: None,
        };
        if M::is_sparse() {
            let op = ret.rhs();
            let ctx = op.context().clone();
            let n = op.nstates();
            let nparams = op.nparams();

            let non_zeros = ret.context.rhs_state_deps.as_slice();

            let sparsity = M::Sparsity::try_from_indices(n, n, non_zeros.to_vec())
                .expect("invalid sparsity pattern");
            let coloring = JacobianColoring::new(&sparsity, non_zeros, ctx.clone());
            ret.rhs_coloring = Some(coloring);
            ret.rhs_sparsity = Some(sparsity);

            let non_zeros = non_zeros.iter().map(|(i, j)| (*j, *i)).collect::<Vec<_>>();
            let sparsity = M::Sparsity::try_from_indices(n, n, non_zeros.clone())
                .expect("invalid sparsity pattern");
            let coloring = JacobianColoring::new(&sparsity, &non_zeros, ctx.clone());
            ret.rhs_adjoint_sparsity = Some(sparsity);
            ret.rhs_adjoint_coloring = Some(coloring);

            if nparams > 0 && include_sensitivities {
                let non_zeros = ret.context.rhs_input_deps.as_slice();

                let sparsity = M::Sparsity::try_from_indices(n, nparams, non_zeros.to_vec())
                    .expect("invalid sparsity pattern");
                let coloring = JacobianColoring::new(&sparsity, non_zeros, ctx.clone());
                ret.rhs_sens_coloring = Some(coloring);
                ret.rhs_sens_sparsity = Some(sparsity);

                let non_zeros = non_zeros.iter().map(|(i, j)| (*j, *i)).collect::<Vec<_>>();
                let sparsity = M::Sparsity::try_from_indices(nparams, n, non_zeros.clone())
                    .expect("invalid sparsity pattern");
                let coloring = JacobianColoring::new(&sparsity, &non_zeros, ctx.clone());
                ret.rhs_sens_adjoint_sparsity = Some(sparsity);
                ret.rhs_sens_adjoint_coloring = Some(coloring);
            }

            let non_zeros = ret.context.mass_state_deps.as_slice();
            if let Some(op) = ret.mass() {
                let ctx = op.context().clone();
                let sparsity = M::Sparsity::try_from_indices(n, n, non_zeros.to_vec())
                    .expect("invalid sparsity pattern");
                let coloring = JacobianColoring::new(&sparsity, non_zeros, op.context().clone());
                ret.mass_coloring = Some(coloring);
                ret.mass_sparsity = Some(sparsity);

                let non_zeros = non_zeros.iter().map(|(i, j)| (*j, *i)).collect::<Vec<_>>();
                let sparsity = M::Sparsity::try_from_indices(n, n, non_zeros.clone())
                    .expect("invalid sparsity pattern");
                let coloring = JacobianColoring::new(&sparsity, &non_zeros, ctx);
                ret.mass_transpose_sparsity = Some(sparsity);
                ret.mass_transpose_coloring = Some(coloring);
            }
        }
        ret
    }

    /// Set the active DiffSL model index together with parameters.
    ///
    /// This updates the compiler input block and then recomputes constants via `set_u0`.
    pub fn set_params_and_model(&mut self, p: &M::V, model_index: u32) {
        self.context.model_index = model_index;
        self.context.compiler.set_inputs(
            p.as_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
            self.context.model_index,
        );
        let mut dummy = M::V::zeros(self.context.nstates, self.context.ctx.clone());
        self.context.compiler.set_u0(
            dummy.as_mut_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
        );
    }
}

#[cfg(feature = "diffsl-external-dynamic")]
impl<M: MatrixHost<T: DiffSlScalar>> DiffSl<M, ExternalDynModule<M::T>> {
    /// Create a `DiffSl` instance using externally-provided functions & sparsity patterns.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the external dynamic library
    /// * `ctx` - The computational context for vector and matrix operations (e.g., CPU, GPU)
    /// * `rhs_state_deps` - Sparsity pattern for the RHS Jacobian (∂f/∂y) as pairs (row, col). Can be empty if M is dense.
    /// * `rhs_input_deps` - Sparsity pattern for the RHS sensitivity matrix (∂f/∂p) as pairs (row, col). Can be empty if M is dense or if there are no parameters.
    /// * `mass_state_deps` - Sparsity pattern for the mass matrix Jacobian (∂M/∂y) as pairs (row, col). Can be empty if there is no mass matrix or if M is dense.
    /// * `include_sensitivities` - Whether to set up sparsity patterns for sensitivity computations.
    ///   If `true`, enables forward and adjoint sensitivity analysis. Set to `false` to skip
    ///   sensitivity setup for better memory efficiency when sensitivities are not needed.
    ///
    /// # Returns
    ///
    /// A new `DiffSl` instance with Jacobian colorings configured for efficient matrix computation,
    /// or an error if the context creation fails.
    pub fn from_external_dynamic(
        path: impl Into<PathBuf>,
        ctx: M::C,
        rhs_state_deps: Vec<(usize, usize)>,
        rhs_input_deps: Vec<(usize, usize)>,
        mass_state_deps: Vec<(usize, usize)>,
        include_sensitivities: bool,
    ) -> Result<Self, DiffsolError> {
        let context = DiffSlContext::<M, ExternalDynModule<M::T>>::new_external_dynamic(
            path,
            1,
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            ctx,
        )?;
        Ok(Self::from_context(context, include_sensitivities))
    }
}

#[cfg(feature = "diffsl-external")]
impl<M: MatrixHost<T: DiffSlScalar + ExternSymbols>> DiffSl<M, ExternalModule<M::T>> {
    /// Create a `DiffSl` instance using externally-provided functions & sparsity patterns.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The computational context for vector and matrix operations (e.g., CPU, GPU)
    /// * `rhs_state_deps` - Sparsity pattern for the RHS Jacobian (∂f/∂y) as pairs (row, col). Can be empty if M is dense.
    /// * `rhs_input_deps` - Sparsity pattern for the RHS sensitivity matrix (∂f/∂p) as pairs (row, col). Can be empty if M is dense or if there are no parameters.
    /// * `mass_state_deps` - Sparsity pattern for the mass matrix Jacobian (∂M/∂y) as pairs (row, col). Can be empty if there is no mass matrix or if M is dense.
    /// * `include_sensitivities` - Whether to set up sparsity patterns for sensitivity computations.
    ///   If `true`, enables forward and adjoint sensitivity analysis. Set to `false` to skip
    ///   sensitivity setup for better memory efficiency when sensitivities are not needed.
    ///
    /// # Returns
    ///
    /// A new `DiffSl` instance with Jacobian colorings configured for efficient matrix computation,
    /// or an error if the context creation fails.
    pub fn from_external(
        ctx: M::C,
        rhs_state_deps: Vec<(usize, usize)>,
        rhs_input_deps: Vec<(usize, usize)>,
        mass_state_deps: Vec<(usize, usize)>,
        include_sensitivities: bool,
    ) -> Result<Self, DiffsolError> {
        let context = DiffSlContext::<M, ExternalModule<M::T>>::new_external(
            1,
            rhs_state_deps,
            rhs_input_deps,
            mass_state_deps,
            ctx,
        )?;
        Ok(Self::from_context(context, include_sensitivities))
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModuleJit + CodegenModuleCompile> DiffSl<M, CG> {
    /// Compile DiffSL code into ODE equations.
    ///
    /// This is a convenience function that creates a new `DiffSlContext` from the provided code
    /// and then calls `from_context` to create the `DiffSl` instance. For more control over
    /// the compilation process (e.g., number of threads), create the context directly using
    /// `DiffSlContext::new` and then call `from_context`.
    ///
    /// # Arguments
    ///
    /// * `code` - The DiffSL code defining the ODE system
    /// * `ctx` - The context for creating vectors and matrices (typically `Default::default()`)
    /// * `include_sensitivities` - Whether to extract sparsity patterns for sensitivity computations.
    ///   Set to `true` if you plan to compute sensitivities or adjoints.
    ///
    /// # Returns
    ///
    /// A new `DiffSl` instance that implements `OdeEquations` and can be used with ODE solvers.
    ///
    /// # Errors
    ///
    /// Returns an error if the DiffSL code cannot be parsed or compiled.
    pub fn compile(
        code: &str,
        ctx: M::C,
        include_sensitivities: bool,
    ) -> Result<Self, DiffsolError> {
        let context = DiffSlContext::<M, CG>::new(code, 1, ctx)?;
        Ok(Self::from_context(context, include_sensitivities))
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule + CodegenModuleEmit> DiffSl<M, CG> {
    fn to_external_object(&self) -> Result<DiffSlExternalObject, DiffsolError> {
        let object = self
            .context
            .compiler
            .module()
            .to_object()
            .map_err(|e| DiffsolError::DiffslCompilerError(e.to_string()))?;
        Ok(DiffSlExternalObject {
            scalar_type: diffsl_external_scalar_type::<M::T>()?,
            object,
            rhs_state_deps: self.context.rhs_state_deps.clone(),
            rhs_input_deps: self.context.rhs_input_deps.clone(),
            mass_state_deps: self.context.mass_state_deps.clone(),
            include_sensitivities: self.include_sensitivities,
        })
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> DiffSl<M, CG>
where
    CG: CodegenModuleLink + CodegenModuleJit,
{
    fn from_external_object(
        external_object: DiffSlExternalObject,
        ctx: M::C,
    ) -> Result<Self, DiffsolError> {
        let expected_scalar_type = diffsl_external_scalar_type::<M::T>()?;
        if external_object.scalar_type != expected_scalar_type {
            return Err(DiffsolError::Other(format!(
                "DiffSl external object scalar type mismatch: object is {:?}, requested {:?}",
                external_object.scalar_type, expected_scalar_type
            )));
        }
        let context = DiffSlContext::<M, CG>::new_from_object(
            external_object.object,
            1,
            external_object.rhs_state_deps,
            external_object.rhs_input_deps,
            external_object.mass_state_deps,
            ctx,
        )?;
        Ok(Self::from_context(
            context,
            external_object.include_sensitivities,
        ))
    }
}

#[cfg(feature = "diffsl-llvm")]
impl<M: MatrixHost<T: DiffSlScalar>> Serialize for DiffSl<M, crate::LlvmModule> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_external_object()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<M: MatrixHost<T: DiffSlScalar>> Serialize for DiffSl<M, ObjectModule> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_external_object()
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de, M: MatrixHost<T: DiffSlScalar>> Deserialize<'de> for DiffSl<M, ObjectModule> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let payload = DiffSlExternalObject::deserialize(deserializer)?;
        Self::from_external_object(payload, M::C::default()).map_err(serde::de::Error::custom)
    }
}

pub struct DiffSlRoot<'a, M: Matrix<T: DiffSlScalar>, CG: CodegenModule>(&'a DiffSl<M, CG>);
pub struct DiffSlReset<'a, M: Matrix<T: DiffSlScalar>, CG: CodegenModule>(&'a DiffSl<M, CG>);
pub struct DiffSlOut<'a, M: Matrix<T: DiffSlScalar>, CG: CodegenModule>(&'a DiffSl<M, CG>);
pub struct DiffSlRhs<'a, M: Matrix<T: DiffSlScalar>, CG: CodegenModule>(&'a DiffSl<M, CG>);
pub struct DiffSlMass<'a, M: Matrix<T: DiffSlScalar>, CG: CodegenModule>(&'a DiffSl<M, CG>);
pub struct DiffSlInit<'a, M: Matrix<T: DiffSlScalar>, CG: CodegenModule>(&'a DiffSl<M, CG>);

macro_rules! impl_op_for_diffsl {
    ($name:ident) => {
        impl<M: Matrix<T: DiffSlScalar>, CG: CodegenModule> Op for $name<'_, M, CG> {
            type M = M;
            type T = M::T;
            type V = M::V;
            type C = M::C;

            fn nstates(&self) -> usize {
                self.0.context.nstates
            }
            #[allow(clippy::misnamed_getters)]
            fn nout(&self) -> usize {
                self.0.context.nstates
            }
            fn nparams(&self) -> usize {
                self.0.context.nparams
            }
            fn context(&self) -> &Self::C {
                &self.0.context.ctx
            }
        }
    };
}

impl_op_for_diffsl!(DiffSlRhs);
impl_op_for_diffsl!(DiffSlMass);

impl<M: Matrix<T: DiffSlScalar>, CG: CodegenModule> Op for DiffSlInit<'_, M, CG> {
    type M = M;
    type T = M::T;
    type V = M::V;
    type C = M::C;

    fn nstates(&self) -> usize {
        self.0.context.nstates
    }
    #[allow(clippy::misnamed_getters)]
    fn nout(&self) -> usize {
        self.0.context.nstates
    }
    fn nparams(&self) -> usize {
        self.0.context.nparams
    }
    fn context(&self) -> &Self::C {
        &self.0.context.ctx
    }
}

impl<M: Matrix<T: DiffSlScalar>, CG: CodegenModule> Op for DiffSlRoot<'_, M, CG> {
    type M = M;
    type T = M::T;
    type V = M::V;
    type C = M::C;

    fn nstates(&self) -> usize {
        self.0.context.nstates
    }
    #[allow(clippy::misnamed_getters)]
    fn nout(&self) -> usize {
        self.0.context.nroots
    }
    fn nparams(&self) -> usize {
        self.0.context.nparams
    }
    fn context(&self) -> &Self::C {
        &self.0.context.ctx
    }
}

impl<M: Matrix<T: DiffSlScalar>, CG: CodegenModule> Op for DiffSlReset<'_, M, CG> {
    type M = M;
    type T = M::T;
    type V = M::V;
    type C = M::C;

    fn nstates(&self) -> usize {
        self.0.context.nstates
    }
    #[allow(clippy::misnamed_getters)]
    fn nout(&self) -> usize {
        self.0.context.nstates
    }
    fn nparams(&self) -> usize {
        self.0.context.nparams
    }
    fn context(&self) -> &Self::C {
        &self.0.context.ctx
    }
}

impl<M: Matrix<T: DiffSlScalar>, CG: CodegenModule> Op for DiffSlOut<'_, M, CG> {
    type M = M;
    type T = M::T;
    type V = M::V;
    type C = M::C;

    fn nstates(&self) -> usize {
        self.0.context.nstates
    }
    fn nout(&self) -> usize {
        self.0.context.nout
    }
    fn nparams(&self) -> usize {
        self.0.context.nparams
    }
    fn context(&self) -> &Self::C {
        &self.0.context.ctx
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> ConstantOp for DiffSlInit<'_, M, CG> {
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        self.0.context.compiler.set_u0(
            y.as_mut_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
        );
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> ConstantOpSens for DiffSlInit<'_, M, CG> {
    fn sens_mul_inplace(&self, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.0.context.compiler.set_inputs(
            v.as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            self.0.context.model_index,
        );
        self.0.context.compiler.set_u0_sgrad(
            self.0.context.tmp.borrow().as_slice(),
            y.as_mut_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
        );
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> ConstantOpSensAdjoint
    for DiffSlInit<'_, M, CG>
{
    fn sens_transpose_mul_inplace(&self, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        // copy v to tmp2
        let mut tmp2 = self.0.context.tmp2.borrow_mut();
        tmp2.copy_from(v);
        // zero out sens_data
        self.0.context.sens_data.borrow_mut().fill(M::T::zero());
        self.0.context.compiler.set_u0_rgrad(
            self.0.context.tmp.borrow().as_slice(),
            tmp2.as_mut_slice(),
            self.0.context.data.borrow().as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
        );
        self.0.context.compiler.get_inputs(
            y.as_mut_slice(),
            self.0.context.sens_data.borrow().as_slice(),
        );
        // negate y
        y.mul_assign(Scale(-M::T::one()));
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOp for DiffSlRoot<'_, M, CG> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.0.context.compiler.calc_stop(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpJacobian
    for DiffSlRoot<'_, M, CG>
{
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let stop = self.0.context.tmp_root.borrow();
        self.0.context.compiler.calc_stop_grad(
            t,
            x.as_slice(),
            v.as_slice(),
            self.0.context.data.borrow().as_slice(),
            self.0.context.ddata.borrow_mut().as_mut_slice(),
            stop.as_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpAdjoint
    for DiffSlRoot<'_, M, CG>
{
    fn jac_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let stop = self.0.context.tmp_root.borrow();
        let mut tmp2_root = self.0.context.tmp2_root.borrow_mut();
        tmp2_root.copy_from(v);
        self.0.context.ddata.borrow_mut().fill(M::T::zero());
        y.fill(M::T::zero());
        self.0.context.compiler.calc_stop_rgrad(
            t,
            x.as_slice(),
            y.as_mut_slice(),
            self.0.context.data.borrow().as_slice(),
            self.0.context.ddata.borrow_mut().as_mut_slice(),
            stop.as_slice(),
            tmp2_root.as_mut_slice(),
        );
        y.mul_assign(Scale(-M::T::one()));
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpSens for DiffSlRoot<'_, M, CG> {
    fn sens_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let stop = self.0.context.tmp_root.borrow();
        self.0.context.compiler.set_inputs(
            v.as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            self.0.context.model_index,
        );
        self.0.context.compiler.calc_stop_sgrad(
            t,
            x.as_slice(),
            self.0.context.data.borrow().as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            stop.as_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpSensAdjoint
    for DiffSlRoot<'_, M, CG>
{
    fn sens_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let stop = self.0.context.tmp_root.borrow();
        let mut tmp2_root = self.0.context.tmp2_root.borrow_mut();
        tmp2_root.copy_from(v);
        self.0.context.sens_data.borrow_mut().fill(M::T::zero());
        self.0.context.compiler.calc_stop_srgrad(
            t,
            x.as_slice(),
            self.0.context.data.borrow().as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            stop.as_slice(),
            tmp2_root.as_mut_slice(),
        );
        self.0.context.compiler.get_inputs(
            y.as_mut_slice(),
            self.0.context.sens_data.borrow().as_slice(),
        );
        y.mul_assign(Scale(-M::T::one()));
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOp for DiffSlReset<'_, M, CG> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.0.context.compiler.reset(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpJacobian
    for DiffSlReset<'_, M, CG>
{
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.0.context.ddata.borrow_mut().fill(M::T::zero());
        let tmp = self.0.context.tmp.borrow();
        self.0.context.compiler.reset_grad(
            t,
            x.as_slice(),
            v.as_slice(),
            self.0.context.data.borrow_mut().as_slice(),
            self.0.context.ddata.borrow_mut().as_mut_slice(),
            tmp.as_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpAdjoint
    for DiffSlReset<'_, M, CG>
{
    fn jac_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        // copy v to tmp2
        let mut tmp2 = self.0.context.tmp2.borrow_mut();
        tmp2.copy_from(v);
        // zero out ddata
        self.0.context.ddata.borrow_mut().fill(M::T::zero());
        // zero y
        y.fill(M::T::zero());
        self.0.context.compiler.reset_rgrad(
            t,
            x.as_slice(),
            y.as_mut_slice(),
            self.0.context.data.borrow().as_slice(),
            self.0.context.ddata.borrow_mut().as_mut_slice(),
            self.0.context.tmp.borrow().as_slice(),
            tmp2.as_mut_slice(),
        );
        // negate y
        y.mul_assign(Scale(-M::T::one()));
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpSens for DiffSlReset<'_, M, CG> {
    fn sens_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let tmp = self.0.context.tmp.borrow();
        self.0.context.compiler.set_inputs(
            v.as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            self.0.context.model_index,
        );
        self.0.context.compiler.reset_sgrad(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            tmp.as_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpSensAdjoint
    for DiffSlReset<'_, M, CG>
{
    fn sens_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let tmp = self.0.context.tmp.borrow();
        // copy v to tmp2
        let mut tmp2 = self.0.context.tmp2.borrow_mut();
        tmp2.copy_from(v);
        // zero out sens_data
        self.0.context.sens_data.borrow_mut().fill(M::T::zero());
        self.0.context.compiler.reset_srgrad(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            tmp.as_slice(),
            tmp2.as_mut_slice(),
        );
        // get inputs
        self.0.context.compiler.get_inputs(
            y.as_mut_slice(),
            self.0.context.sens_data.borrow().as_slice(),
        );
        // negate y
        y.mul_assign(Scale(-M::T::one()));
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOp for DiffSlOut<'_, M, CG> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.0.context.compiler.calc_out(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpJacobian
    for DiffSlOut<'_, M, CG>
{
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        // init ddata with all zero except for out
        let mut ddata = self.0.context.ddata.borrow_mut();
        ddata.fill(M::T::zero());
        self.0.context.compiler.calc_out_grad(
            t,
            x.as_slice(),
            v.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            ddata.as_mut_slice(),
            self.0.context.tmp_out.borrow().as_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpAdjoint
    for DiffSlOut<'_, M, CG>
{
    fn jac_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        // init ddata with all zero except for out
        let mut ddata = self.0.context.ddata.borrow_mut();
        ddata.fill(M::T::zero());
        let mut tmp2_out = self.0.context.tmp2_out.borrow_mut();
        tmp2_out.copy_from(v);
        // zero y
        y.fill(M::T::zero());
        self.0.context.compiler.calc_out_rgrad(
            t,
            x.as_slice(),
            y.as_mut_slice(),
            self.0.context.data.borrow_mut().as_slice(),
            ddata.as_mut_slice(),
            self.0.context.tmp_out.borrow().as_slice(),
            tmp2_out.as_mut_slice(),
        );
        // negate y
        y.mul_assign(Scale(-M::T::one()));
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpSens for DiffSlOut<'_, M, CG> {
    fn sens_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        // set inputs for sens_data
        self.0.context.compiler.set_inputs(
            v.as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            self.0.context.model_index,
        );
        self.0.context.compiler.calc_out_sgrad(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            self.0.context.tmp_out.borrow().as_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpSensAdjoint
    for DiffSlOut<'_, M, CG>
{
    fn sens_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let mut sens_data = self.0.context.sens_data.borrow_mut();
        // set outputs for sens_data (zero everything except for out)
        sens_data.fill(M::T::zero());
        let mut tmp2_out = self.0.context.tmp2_out.borrow_mut();
        tmp2_out.copy_from(v);
        self.0.context.compiler.calc_out_srgrad(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            sens_data.as_mut_slice(),
            self.0.context.tmp_out.borrow().as_slice(),
            tmp2_out.as_mut_slice(),
        );
        // set y to the result in inputs
        self.0
            .context
            .compiler
            .get_inputs(y.as_mut_slice(), sens_data.as_slice());
        // negate y
        y.mul_assign(Scale(-M::T::one()));
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOp for DiffSlRhs<'_, M, CG> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.0.context.compiler.rhs(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpJacobian
    for DiffSlRhs<'_, M, CG>
{
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.0.context.ddata.borrow_mut().fill(M::T::zero());
        let tmp = self.0.context.tmp.borrow();
        self.0.context.compiler.rhs_grad(
            t,
            x.as_slice(),
            v.as_slice(),
            self.0.context.data.borrow_mut().as_slice(),
            self.0.context.ddata.borrow_mut().as_mut_slice(),
            tmp.as_slice(),
            y.as_mut_slice(),
        );
    }

    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.0.rhs_coloring {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            self._default_jacobian_inplace(x, t, y);
        }
    }
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.0.rhs_sparsity.clone()
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpAdjoint
    for DiffSlRhs<'_, M, CG>
{
    fn jac_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        // copy v to tmp2
        let mut tmp2 = self.0.context.tmp2.borrow_mut();
        tmp2.copy_from(v);
        // zero out ddata
        self.0.context.ddata.borrow_mut().fill(M::T::zero());
        // zero y
        y.fill(M::T::zero());
        self.0.context.compiler.rhs_rgrad(
            t,
            x.as_slice(),
            y.as_mut_slice(),
            self.0.context.data.borrow().as_slice(),
            self.0.context.ddata.borrow_mut().as_mut_slice(),
            self.0.context.tmp.borrow().as_slice(),
            tmp2.as_mut_slice(),
        );
        // negate y
        y.mul_assign(Scale(-M::T::one()));
    }
    fn adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        // if we have a rhs_coloring and no rhs_adjoint_coloring, user has not called prep_adjoint
        // fail here
        if self.0.rhs_coloring.is_some() && self.0.rhs_adjoint_coloring.is_none() {
            panic!("Adjoint not prepared. Call prep_adjoint before calling adjoint_inplace");
        }
        if let Some(coloring) = &self.0.rhs_adjoint_coloring {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            self._default_adjoint_inplace(x, t, y);
        }
    }
    fn adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.0.rhs_adjoint_sparsity.clone()
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpSens for DiffSlRhs<'_, M, CG> {
    fn sens_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let tmp = self.0.context.tmp.borrow();
        self.0.context.compiler.set_inputs(
            v.as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            self.0.context.model_index,
        );
        self.0.context.compiler.rhs_sgrad(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            tmp.as_slice(),
            y.as_mut_slice(),
        );
    }
    fn sens_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.0.rhs_sens_coloring {
            coloring.sens_inplace(self, x, t, y);
        } else {
            self._default_sens_inplace(x, t, y);
        }
    }
    fn sens_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.0.rhs_sens_sparsity.clone()
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> NonLinearOpSensAdjoint
    for DiffSlRhs<'_, M, CG>
{
    fn sens_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        // todo: would rhs_srgrad ever use rr? I don't think so, but need to check
        let tmp = self.0.context.tmp.borrow();
        // copy v to tmp2
        let mut tmp2 = self.0.context.tmp2.borrow_mut();
        tmp2.copy_from(v);
        // zero out sens_data
        self.0.context.sens_data.borrow_mut().fill(M::T::zero());
        self.0.context.compiler.rhs_srgrad(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            tmp.as_slice(),
            tmp2.as_mut_slice(),
        );
        // get inputs
        self.0.context.compiler.get_inputs(
            y.as_mut_slice(),
            self.0.context.sens_data.borrow().as_slice(),
        );
        // negate y
        y.mul_assign(Scale(-M::T::one()));
    }
    fn sens_adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.0.rhs_sens_adjoint_coloring {
            coloring.sens_adjoint_inplace(self, x, t, y);
        } else {
            self._default_adjoint_inplace(x, t, y);
        }
    }
    fn sens_adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.0.rhs_sens_adjoint_sparsity.clone()
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> LinearOp for DiffSlMass<'_, M, CG> {
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        let mut tmp = self.0.context.tmp.borrow_mut();
        self.0.context.compiler.mass(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            tmp.as_mut_slice(),
        );

        // y = tmp + beta * y
        y.axpy(M::T::one(), &tmp, beta);
    }

    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.0.mass_coloring {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_matrix_inplace(t, y);
        }
    }
    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.0.mass_sparsity.clone()
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> LinearOpTranspose
    for DiffSlMass<'_, M, CG>
{
    fn gemv_transpose_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        // scale y by beta
        y.mul_assign(Scale(beta));

        // copy x to tmp
        let mut tmp = self.0.context.tmp.borrow_mut();
        tmp.copy_from(x);

        // zero out ddata
        self.0.context.ddata.borrow_mut().fill(M::T::zero());

        // y += M^T x + beta * y
        self.0.context.compiler.mass_rgrad(
            t,
            y.as_mut_slice(),
            self.0.context.data.borrow_mut().as_slice(),
            self.0.context.ddata.borrow_mut().as_mut_slice(),
            tmp.as_mut_slice(),
        );
    }

    fn transpose_inplace(&self, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.0.mass_transpose_coloring {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_matrix_inplace(t, y);
        }
    }
    fn transpose_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.0.mass_transpose_sparsity.clone()
    }
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> Op for DiffSl<M, CG> {
    type M = M;
    type T = M::T;
    type V = M::V;
    type C = M::C;

    fn nstates(&self) -> usize {
        self.context.nstates
    }
    fn nout(&self) -> usize {
        if self.context.has_out {
            self.context.nout
        } else {
            self.context.nstates
        }
    }
    fn nparams(&self) -> usize {
        self.context.nparams
    }
    fn context(&self) -> &Self::C {
        &self.context.ctx
    }
}

impl<'a, M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> OdeEquationsRef<'a> for DiffSl<M, CG> {
    type Mass = DiffSlMass<'a, M, CG>;
    type Rhs = DiffSlRhs<'a, M, CG>;
    type Root = DiffSlRoot<'a, M, CG>;
    type Init = DiffSlInit<'a, M, CG>;
    type Out = DiffSlOut<'a, M, CG>;
    type Reset = DiffSlReset<'a, M, CG>;
}

impl<M: MatrixHost<T: DiffSlScalar>, CG: CodegenModule> OdeEquations for DiffSl<M, CG> {
    fn rhs(&self) -> DiffSlRhs<'_, M, CG> {
        DiffSlRhs(self)
    }

    fn mass(&self) -> Option<DiffSlMass<'_, M, CG>> {
        self.context.has_mass.then_some(DiffSlMass(self))
    }

    fn root(&self) -> Option<DiffSlRoot<'_, M, CG>> {
        self.context.has_root.then_some(DiffSlRoot(self))
    }

    fn init(&self) -> DiffSlInit<'_, M, CG> {
        DiffSlInit(self)
    }

    fn out(&self) -> Option<DiffSlOut<'_, M, CG>> {
        self.context.has_out.then_some(DiffSlOut(self))
    }

    fn reset(&self) -> Option<DiffSlReset<'_, M, CG>> {
        self.context.has_reset.then_some(DiffSlReset(self))
    }

    fn set_params(&mut self, p: &Self::V) {
        // `set_params` preserves the current model index.
        self.set_params_and_model(p, self.context.model_index);
    }

    fn set_model_index(&mut self, m: usize) {
        self.context.model_index = m as u32;
        let mut p = M::V::zeros(self.nparams(), self.context.ctx.clone());
        self.get_params(&mut p);
        self.set_params_and_model(&p, self.context.model_index);
    }

    fn get_params(&self, p: &mut Self::V) {
        self.context
            .compiler
            .get_inputs(p.as_mut_slice(), self.context.data.borrow().as_slice());
    }
}

#[cfg(test)]
mod tests {
    use diffsl::execution::{
        module::{CodegenModuleCompile, CodegenModuleJit},
        scalar::Scalar as DiffSlScalar,
    };
    use diffsl::ObjectModule;

    use crate::Scalar;
    use crate::{
        error::DiffsolError, matrix::MatrixRef, ConstantOp, Context, DefaultDenseMatrix,
        DefaultSolver, DenseMatrix, LinearOp, Matrix, NonLinearOp, NonLinearOpAdjoint,
        NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint, OdeBuilder, OdeEquations,
        OdeSolverMethod, Vector, VectorHost, VectorRef, VectorView,
    };

    use super::{DiffSl, DiffSlContext};
    use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
    use paste::paste;
    use serde_json;

    /// Macro to generate test functions for all combinations of backend (cranelift/llvm) and scalar type (f32/f64)
    ///
    /// Usage: `generate_tests!(test_name, generic_test_function);`
    ///
    /// This will generate 4 test functions:
    /// - {test_name}_cranelift_f64
    /// - {test_name}_cranelift_f32
    /// - {test_name}_llvm_f64
    /// - {test_name}_llvm_f32
    ///
    /// Example:
    /// ```
    /// fn my_test<M: CodegenModuleCompile + CodegenModuleJit, T: Scalar>() { ... }
    /// generate_tests!(my_test);
    /// ```
    macro_rules! generate_tests {
        ($test_fn:ident) => {
            generate_tests!(@impl $test_fn, cranelift_dense_f64, crate::CraneliftJitModule, crate::NalgebraMat<f64>, "diffsl-cranelift");
            generate_tests!(@impl $test_fn, cranelift_sparse_f64, crate::CraneliftJitModule, crate::FaerSparseMat<f64>, "diffsl-cranelift");
            generate_tests!(@impl $test_fn, cranelift_dense_f32, crate::CraneliftJitModule, crate::NalgebraMat<f32>, "diffsl-cranelift");
            generate_tests!(@impl $test_fn, cranelift_sparse_f32, crate::CraneliftJitModule, crate::FaerSparseMat<f32>, "diffsl-cranelift");
            generate_tests!(@impl $test_fn, llvm_dense_f64, crate::LlvmModule, crate::NalgebraMat<f64>, "diffsl-llvm");
            generate_tests!(@impl $test_fn, llvm_sparse_f64, crate::LlvmModule, crate::FaerSparseMat<f64>, "diffsl-llvm");
            generate_tests!(@impl $test_fn, llvm_dense_f32, crate::LlvmModule, crate::NalgebraMat<f32>, "diffsl-llvm");
            generate_tests!(@impl $test_fn, llvm_sparse_f32, crate::LlvmModule, crate::FaerSparseMat<f32>, "diffsl-llvm");
        };
        (@impl $test_fn:ident, $variant:ident, $module:ty, $matrix:ty, $feature:literal) => {
            paste! {
                #[cfg(feature = $feature)]
                #[test]
                fn [<$test_fn _ $variant>]() {
                    $test_fn::<$module, $matrix>();
                }
            }
        };
    }

    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    generate_tests!(diffsl_logistic_growth);
    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    generate_tests!(diffsl_logistic_growth_with_model_index);
    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    generate_tests!(diffsl_reset_call_and_jac_mul);
    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    generate_tests!(diffsl_context_handles_thread_modes);
    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    generate_tests!(diffsl_context_reports_parser_and_compiler_errors);
    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    generate_tests!(diffsl_root_and_output_operators_work);

    // Sensitivity and reverse-mode (adjoint) require LLVM — Cranelift supports neither.
    macro_rules! generate_tests_llvm_only {
        ($test_fn:ident) => {
            generate_tests!(@impl $test_fn, llvm_dense_f64, crate::LlvmModule, crate::NalgebraMat<f64>, "diffsl-llvm");
            generate_tests!(@impl $test_fn, llvm_sparse_f64, crate::LlvmModule, crate::FaerSparseMat<f64>, "diffsl-llvm");
            generate_tests!(@impl $test_fn, llvm_dense_f32, crate::LlvmModule, crate::NalgebraMat<f32>, "diffsl-llvm");
            generate_tests!(@impl $test_fn, llvm_sparse_f32, crate::LlvmModule, crate::FaerSparseMat<f32>, "diffsl-llvm");
        };
    }

    generate_tests_llvm_only!(diffsl_reset_sens_and_adjoint_gradients);
    generate_tests_llvm_only!(diffsl_root_sens_gradients);
    /// Tests forward evaluation and Jacobian-vector product for DiffSlReset.
    /// Runs on all backends (Cranelift + LLVM).
    ///
    /// Model: reset_i { 2 * y + a, z + a }  with a=3, (y,z)=(3,2), t=0.
    ///   J = d(reset)/d(x) = [[2, 0], [0, 1]]
    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    fn diffsl_reset_call_and_jac_mul<
        CG: CodegenModuleJit + CodegenModuleCompile,
        M: Matrix<V: VectorHost + DefaultDenseMatrix, T: DiffSlScalar> + DefaultSolver,
    >()
    where
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        let text = "
            in { a = 1 }
            u_i {
                y = a,
                z = 2,
            }
            F_i {
                y,
                z,
            }
            reset_i {
                2 * y + a,
                z + a,
            }
            stop_i {
                y - 0.5,
            }
            out_i {
                y,
                z,
            }
        ";

        let ctx = M::C::default();
        let a = M::T::from_f64(3.0).unwrap();
        let p = ctx.vector_from_vec(vec![a]);
        let mut eqn = DiffSl::<M, CG>::compile(text, ctx.clone(), false).unwrap();
        eqn.set_params(&p);

        // x = (y, z) = (a, 2) = (3, 2) after set_params
        let x = eqn.init().call(M::T::zero());
        let t = M::T::zero();
        let reset_op = eqn.reset().expect("model must have a reset operator");

        // reset(x, t) = [2*3+3, 2+3] = [9, 5]
        let reset_val = reset_op.call(&x, t);
        let reset_expected = ctx.vector_from_vec(vec![
            M::T::from_f64(9.0).unwrap(),
            M::T::from_f64(5.0).unwrap(),
        ]);
        reset_val.assert_eq_st(&reset_expected, M::T::from_f64(1e-10).unwrap());

        // jac_mul: J*v, J=[[2,0],[0,1]], v=[3,-1] => [6,-1]
        let v = ctx.vector_from_vec(vec![M::T::from_f64(3.0).unwrap(), -M::T::one()]);
        let mut y = ctx.vector_from_vec(vec![M::T::zero(), M::T::zero()]);
        reset_op.jac_mul_inplace(&x, t, &v, &mut y);
        let jac_mul_expected =
            ctx.vector_from_vec(vec![M::T::from_f64(6.0).unwrap(), -M::T::one()]);
        y.assert_eq_st(&jac_mul_expected, M::T::from_f64(1e-10).unwrap());
    }

    /// Tests sensitivity and adjoint gradient products for DiffSlReset.
    /// Requires LLVM — Cranelift does not compile sensitivity or reverse-mode autograd.
    ///
    /// Model: reset_i { 2 * y + a, z + a }  with a=3, (y,z)=(3,2), t=0.
    ///   d(reset)/d(a) = [1, 1]
    ///   J^T = [[2, 0], [0, 1]] (diagonal, same as J)
    ///
    /// Note: jac_transpose_mul and sens_transpose_mul return negated values
    ///       (same convention as rhs adjoint).
    #[allow(dead_code)]
    fn diffsl_reset_sens_and_adjoint_gradients<
        CG: CodegenModuleJit + CodegenModuleCompile,
        M: Matrix<V: VectorHost + DefaultDenseMatrix, T: DiffSlScalar> + DefaultSolver,
    >()
    where
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        let text = "
            in { a = 1 }
            u_i {
                y = a,
                z = 2,
            }
            F_i {
                y,
                z,
            }
            reset_i {
                2 * y + a,
                z + a,
            }
            stop_i {
                y - 0.5,
            }
            out_i {
                y,
                z,
            }
        ";

        let ctx = M::C::default();
        let a = M::T::from_f64(3.0).unwrap();
        let p = ctx.vector_from_vec(vec![a]);
        let mut eqn = DiffSl::<M, CG>::compile(text, ctx.clone(), false).unwrap();
        eqn.set_params(&p);

        let x = eqn.init().call(M::T::zero());
        let t = M::T::zero();
        let reset_op = eqn.reset().expect("model must have a reset operator");

        let v = ctx.vector_from_vec(vec![M::T::from_f64(3.0).unwrap(), -M::T::one()]);

        // sens_mul: (d_reset/d_a)*vp, d/da=[1,1], vp=[2] => [2,2]
        let vp = ctx.vector_from_vec(vec![M::T::from_f64(2.0).unwrap()]);
        let mut y = ctx.vector_from_vec(vec![M::T::zero(), M::T::zero()]);
        reset_op.sens_mul_inplace(&x, t, &vp, &mut y);
        let sens_expected = ctx.vector_from_vec(vec![
            M::T::from_f64(2.0).unwrap(),
            M::T::from_f64(2.0).unwrap(),
        ]);
        y.assert_eq_st(&sens_expected, M::T::from_f64(1e-10).unwrap());

        // jac_transpose_mul: -J^T*v, J=[[2,0],[0,1]], v=[3,-1] => -[6,-1] = [-6,1]
        let mut y = ctx.vector_from_vec(vec![M::T::zero(), M::T::zero()]);
        reset_op.jac_transpose_mul_inplace(&x, t, &v, &mut y);
        let jac_adj_expected =
            ctx.vector_from_vec(vec![M::T::from_f64(-6.0).unwrap(), M::T::one()]);
        y.assert_eq_st(&jac_adj_expected, M::T::from_f64(1e-10).unwrap());

        // sens_transpose_mul: -(d_reset/d_a)^T*v = -(1*3 + 1*(-1)) = -2
        let mut y_p = ctx.vector_from_vec(vec![M::T::zero()]);
        reset_op.sens_transpose_mul_inplace(&x, t, &v, &mut y_p);
        let sens_adj_expected = ctx.vector_from_vec(vec![M::T::from_f64(-2.0).unwrap()]);
        y_p.assert_eq_st(&sens_adj_expected, M::T::from_f64(1e-10).unwrap());
    }

    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    fn diffsl_logistic_growth<
        CG: CodegenModuleJit + CodegenModuleCompile,
        M: Matrix<V: VectorHost + DefaultDenseMatrix, T: DiffSlScalar> + DefaultSolver,
    >()
    where
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        let text = "
            in_i { r = 1, k = 1 }
            u_i {
                y = 0.1,
                z = 0,
            }
            dudt_i {
                dydt = 0,
                dzdt = 0,
            }
            M_i {
                dydt,
                0,
            }
            F_i {
                (r * y) * (1 - (y / k)),
                (2 * y) - z,
            }
            out_i {
                3 * y,
                4 * z,
            }
        ";

        let k = M::T::one();
        let r = M::T::one();
        let ctx = M::C::default();
        let context = DiffSlContext::<M, CG>::new(text, 1, ctx.clone()).unwrap();
        let p = ctx.vector_from_vec(vec![r, k]);
        let mut eqn = DiffSl::from_context(context, false);
        eqn.set_params(&p);

        // test that the initial values look ok
        let y0 = M::T::from_f64(0.1).unwrap();
        let init = eqn.init().call(M::T::zero());
        let init_expect = ctx.vector_from_vec(vec![y0, M::T::zero()]);
        init.assert_eq_st(&init_expect, M::T::from_f64(1e-10).unwrap());
        let rhs = eqn.rhs().call(&init, M::T::zero());
        let rhs_expect = ctx.vector_from_vec(vec![
            r * y0 * (M::T::one() - y0 / k),
            M::T::from_f64(2.0).unwrap() * y0,
        ]);
        rhs.assert_eq_st(&rhs_expect, M::T::from_f64(1e-10).unwrap());
        let v = ctx.vector_from_vec(vec![M::T::one(), M::T::one()]);
        let rhs_jac = eqn.rhs().jac_mul(&init, M::T::zero(), &v);
        let rhs_jac_expect =
            ctx.vector_from_vec(vec![r * (M::T::one() - y0 / k) - r * y0 / k, M::T::one()]);
        rhs_jac.assert_eq_st(&rhs_jac_expect, M::T::from_f64(1e-10).unwrap());
        let mut mass_y = ctx.vector_from_vec(vec![M::T::zero(), M::T::zero()]);
        let v = ctx.vector_from_vec(vec![M::T::one(), M::T::one()]);
        eqn.mass()
            .unwrap()
            .call_inplace(&v, M::T::zero(), &mut mass_y);
        let mass_y_expect = ctx.vector_from_vec(vec![M::T::one(), M::T::zero()]);
        mass_y.assert_eq_st(&mass_y_expect, M::T::from_f64(1e-10).unwrap());

        // solver a bit and check the state and output
        let atol = 1e-4;
        let rtol = 1e-4;
        let problem = OdeBuilder::<M>::new()
            .p([r.to_f64().unwrap(), k.to_f64().unwrap()])
            .atol([atol])
            .rtol(rtol)
            .build_from_eqn(eqn)
            .unwrap();
        let mut solver = problem.bdf::<<M as DefaultSolver>::LS>().unwrap();
        let t = M::T::one();
        let (ys, ts, _stop_reason) = solver.solve(t).unwrap();
        for (i, t) in ts.iter().enumerate() {
            let y_expect = k / (M::T::one() + (k - y0) * (-r * *t).exp() / y0);
            let z_expect = M::T::from_f64(2.0).unwrap() * y_expect;
            let expected_out = ctx.vector_from_vec(vec![
                M::T::from_f64(3.0).unwrap() * y_expect,
                M::T::from_f64(4.0).unwrap() * z_expect,
            ]);
            ys.column(i).into_owned().assert_eq_norm(
                &expected_out,
                &problem.atol,
                problem.rtol,
                M::T::from_f64(10.0).unwrap(),
            );
        }

        // do it again with some explicit t_evals
        let t_evals = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0];
        let t_evals = t_evals
            .into_iter()
            .map(|t| M::T::from_f64(t).unwrap())
            .collect::<Vec<_>>();
        let mut solver = problem.bdf::<<M as DefaultSolver>::LS>().unwrap();
        let (ys, _stop_reason) = solver.solve_dense(&t_evals).unwrap();
        for (i, t) in t_evals.iter().enumerate() {
            let y_expect = k / (M::T::one() + (k - y0) * (-r * *t).exp() / y0);
            let z_expect = M::T::from_f64(2.0).unwrap() * y_expect;
            let expected_out = ctx.vector_from_vec(vec![
                M::T::from_f64(3.0).unwrap() * y_expect,
                M::T::from_f64(4.0).unwrap() * z_expect,
            ]);
            ys.column(i).into_owned().assert_eq_norm(
                &expected_out,
                &problem.atol,
                problem.rtol,
                M::T::from_f64(10.0).unwrap(),
            );
        }
    }

    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    fn diffsl_context_handles_thread_modes<
        CG: CodegenModuleJit + CodegenModuleCompile,
        M: Matrix<V: VectorHost + DefaultDenseMatrix, T: DiffSlScalar> + DefaultSolver,
    >() {
        let text = "
            in_i { r = 1 }
            u_i { y = 0.1 }
            F_i { r * y }
            out_i { y }
        ";

        for nthreads in [0, 1, 4] {
            let context = DiffSlContext::<M, CG>::new(text, nthreads, M::C::default()).unwrap();
            assert_eq!(context.nstates, 1);
            assert_eq!(context.nparams, 1);
            assert_eq!(context.nout, 1);
            assert!(!context.has_mass);
            assert!(!context.has_root);
            assert!(!context.has_reset);
            assert!(context.has_out);
        }
    }

    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    fn diffsl_context_reports_parser_and_compiler_errors<
        CG: CodegenModuleJit + CodegenModuleCompile,
        M: Matrix<V: VectorHost + DefaultDenseMatrix, T: DiffSlScalar> + DefaultSolver,
    >() {
        let parser_err = match DiffSlContext::<M, CG>::new("this is not diffsl", 1, M::C::default())
        {
            Ok(_) => panic!("expected parser error"),
            Err(err) => err,
        };
        assert!(matches!(parser_err, DiffsolError::DiffslParserError(_)));

        let compiler_err = match DiffSlContext::<M, CG>::new(
            "
                u_i { y = 1 }
                F_i { missing_symbol }
            ",
            1,
            M::C::default(),
        ) {
            Ok(_) => panic!("expected compiler error"),
            Err(err) => err,
        };
        assert!(matches!(compiler_err, DiffsolError::DiffslCompilerError(_)));
    }

    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    fn diffsl_root_and_output_operators_work<
        CG: CodegenModuleJit + CodegenModuleCompile,
        M: Matrix<V: VectorHost + DefaultDenseMatrix, T: DiffSlScalar> + DefaultSolver,
    >()
    where
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        let text = "
            in_i { a = 1 }
            u_i {
                y = a,
                z = 2,
            }
            F_i {
                y,
                z,
            }
            stop_i {
                y - 0.5,
            }
            reset_i {
                2 * y + a,
                z + a,
            }
            out_i {
                3 * y,
                4 * z,
            }
        ";

        let ctx = M::C::default();
        let mut eqn = DiffSl::<M, CG>::compile(text, ctx.clone(), false).unwrap();
        let p = ctx.vector_from_vec(vec![M::T::from_f64(3.0).unwrap()]);
        eqn.set_params(&p);

        let x = eqn.init().call(M::T::zero());
        let root = eqn.root().unwrap().call(&x, M::T::zero());
        root.assert_eq_st(
            &ctx.vector_from_vec(vec![M::T::from_f64(2.5).unwrap()]),
            M::T::from_f64(1e-10).unwrap(),
        );

        let root_op = eqn.root().unwrap();
        let v = ctx.vector_from_vec(vec![M::T::from_f64(2.0).unwrap(), -M::T::one()]);
        let mut root_jvp = ctx.vector_from_vec(vec![M::T::zero()]);
        root_op.jac_mul_inplace(&x, M::T::zero(), &v, &mut root_jvp);
        root_jvp.assert_eq_st(
            &ctx.vector_from_vec(vec![M::T::from_f64(2.0).unwrap()]),
            M::T::from_f64(1e-10).unwrap(),
        );

        let out = eqn.out().unwrap().call(&x, M::T::zero());
        out.assert_eq_st(
            &ctx.vector_from_vec(vec![
                M::T::from_f64(9.0).unwrap(),
                M::T::from_f64(8.0).unwrap(),
            ]),
            M::T::from_f64(1e-10).unwrap(),
        );
    }

    #[allow(dead_code)]
    fn diffsl_root_sens_gradients<
        CG: CodegenModuleJit + CodegenModuleCompile,
        M: Matrix<V: VectorHost + DefaultDenseMatrix, T: DiffSlScalar> + DefaultSolver,
    >()
    where
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        let text = "
            in_i { a = 1 }
            u_i {
                y = a,
                z = 2,
            }
            F_i {
                y,
                z,
            }
            stop_i {
                y + a - 0.5,
            }
        ";

        let ctx = M::C::default();
        let mut eqn = DiffSl::<M, CG>::compile(text, ctx.clone(), false).unwrap();
        let p = ctx.vector_from_vec(vec![M::T::from_f64(3.0).unwrap()]);
        eqn.set_params(&p);

        let x = eqn.init().call(M::T::zero());
        let root_op = eqn.root().expect("model must have a root operator");
        let vp = ctx.vector_from_vec(vec![M::T::from_f64(2.0).unwrap()]);
        let mut y = ctx.vector_from_vec(vec![M::T::zero()]);

        root_op.sens_mul_inplace(&x, M::T::zero(), &vp, &mut y);
        y.assert_eq_st(
            &ctx.vector_from_vec(vec![M::T::from_f64(2.0).unwrap()]),
            M::T::from_f64(1e-10).unwrap(),
        );

        let v = ctx.vector_from_vec(vec![M::T::from_f64(3.0).unwrap()]);
        let mut y_x = ctx.vector_from_vec(vec![M::T::zero(), M::T::zero()]);
        root_op.jac_transpose_mul_inplace(&x, M::T::zero(), &v, &mut y_x);
        y_x.assert_eq_st(
            &ctx.vector_from_vec(vec![-M::T::from_f64(3.0).unwrap(), M::T::zero()]),
            M::T::from_f64(1e-10).unwrap(),
        );

        let mut y_p = ctx.vector_from_vec(vec![M::T::zero()]);
        root_op.sens_transpose_mul_inplace(&x, M::T::zero(), &v, &mut y_p);
        y_p.assert_eq_st(
            &ctx.vector_from_vec(vec![-M::T::from_f64(3.0).unwrap()]),
            M::T::from_f64(1e-10).unwrap(),
        );
    }

    #[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
    fn diffsl_logistic_growth_with_model_index<
        CG: CodegenModuleJit + CodegenModuleCompile,
        M: Matrix<V: VectorHost + DefaultDenseMatrix, T: DiffSlScalar> + DefaultSolver,
    >()
    where
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        let text = "
            r_i {
                1,
                2,
                4,
            }
            u_i {
                y = 0.1,
            }
            reset_i {
                y,
            }
            stop_i {
                y - 0.5,
            }
            F_i {
                r_i[N] * y,
            }
        ";

        let ctx = M::C::default();
        let mut eqn = DiffSl::<M, CG>::compile(text, ctx.clone(), false).unwrap();
        let t = M::T::zero();
        let y = eqn.init().call(t);
        let tol = M::T::from_f64(1e-10).unwrap();
        let one_tenth = M::T::from_f64(0.1).unwrap();
        let p = ctx.vector_from_vec(Vec::<M::T>::new());

        let rhs_model_0 = eqn.rhs().call(&y, t);
        let rhs_model_0_expected =
            ctx.vector_from_vec(vec![M::T::from_f64(1.0).unwrap() * one_tenth]);
        rhs_model_0.assert_eq_st(&rhs_model_0_expected, tol);

        eqn.set_model_index(1);
        let rhs_model_1 = eqn.rhs().call(&y, t);
        let rhs_model_1_expected =
            ctx.vector_from_vec(vec![M::T::from_f64(2.0).unwrap() * one_tenth]);
        rhs_model_1.assert_eq_st(&rhs_model_1_expected, tol);

        eqn.set_model_index(2);
        let rhs_model_2 = eqn.rhs().call(&y, t);
        let rhs_model_2_expected =
            ctx.vector_from_vec(vec![M::T::from_f64(4.0).unwrap() * one_tenth]);
        rhs_model_2.assert_eq_st(&rhs_model_2_expected, tol);

        // set_params preserves the current model index.
        eqn.set_params(&p);
        let rhs_after_set_params = eqn.rhs().call(&y, t);
        rhs_after_set_params.assert_eq_st(&rhs_model_2_expected, tol);
    }

    #[cfg(feature = "diffsl-llvm")]
    fn serialization_test_model() -> &'static str {
        "
            in_i { a = 1, b = 2 }
            u_i {
                y = a,
                z = 2,
            }
            dudt_i {
                dydt = 0,
                dzdt = 0,
            }
            M_i {
                dydt,
                0,
            }
            F_i {
                a * y + b,
                z + a,
            }
            stop_i {
                y + a - 0.5,
            }
            reset_i {
                2 * y + a,
                z + a,
            }
            out_i {
                3 * y,
                4 * z,
            }
        "
    }

    #[cfg(feature = "diffsl-llvm")]
    fn assert_object_roundtrip<M>(include_sensitivities: bool)
    where
        M: Matrix<V: VectorHost + DefaultDenseMatrix, T: DiffSlScalar> + DefaultSolver,
        for<'b> &'b M::V: VectorRef<M::V>,
        for<'b> &'b M: MatrixRef<M>,
    {
        let ctx = M::C::default();
        let p = ctx.vector_from_vec(vec![
            M::T::from_f64(3.0).unwrap(),
            M::T::from_f64(5.0).unwrap(),
        ]);
        let mut compiled = DiffSl::<M, crate::LlvmModule>::compile(
            serialization_test_model(),
            ctx.clone(),
            include_sensitivities,
        )
        .unwrap();
        compiled.set_params(&p);
        let rhs_state_deps = compiled.context.rhs_state_deps.clone();
        let rhs_input_deps = compiled.context.rhs_input_deps.clone();
        let mass_state_deps = compiled.context.mass_state_deps.clone();

        let t = M::T::zero();
        let x_compiled = compiled.init().call(t);
        let rhs_compiled = compiled.rhs().call(&x_compiled, t);
        let v = ctx.vector_from_vec(vec![M::T::one(), M::T::one()]);
        let mut mass_compiled = ctx.vector_from_vec(vec![M::T::zero(), M::T::zero()]);
        compiled
            .mass()
            .unwrap()
            .call_inplace(&v, t, &mut mass_compiled);
        let root_compiled = compiled.root().unwrap().call(&x_compiled, t);
        let out_compiled = compiled.out().unwrap().call(&x_compiled, t);
        let reset_compiled = compiled.reset().unwrap().call(&x_compiled, t);
        let external_object = compiled.to_external_object().unwrap();
        let mut imported =
            DiffSl::<M, ObjectModule>::from_external_object(external_object, ctx.clone()).unwrap();
        imported.set_params(&p);

        let x_imported = imported.init().call(t);
        x_imported.assert_eq_st(&x_compiled, M::T::from_f64(1e-10).unwrap());
        let rhs_imported = imported.rhs().call(&x_imported, t);
        rhs_imported.assert_eq_st(&rhs_compiled, M::T::from_f64(1e-10).unwrap());
        let mut mass_imported = ctx.vector_from_vec(vec![M::T::zero(), M::T::zero()]);
        imported
            .mass()
            .unwrap()
            .call_inplace(&v, t, &mut mass_imported);
        mass_imported.assert_eq_st(&mass_compiled, M::T::from_f64(1e-10).unwrap());
        let root_imported = imported.root().unwrap().call(&x_imported, t);
        root_imported.assert_eq_st(&root_compiled, M::T::from_f64(1e-10).unwrap());
        let out_imported = imported.out().unwrap().call(&x_imported, t);
        out_imported.assert_eq_st(&out_compiled, M::T::from_f64(1e-10).unwrap());
        let reset_imported = imported.reset().unwrap().call(&x_imported, t);
        reset_imported.assert_eq_st(&reset_compiled, M::T::from_f64(1e-10).unwrap());

        assert_eq!(imported.context.rhs_state_deps, rhs_state_deps);
        assert_eq!(imported.context.rhs_input_deps, rhs_input_deps);
        assert_eq!(imported.context.mass_state_deps, mass_state_deps);
        assert_eq!(imported.include_sensitivities, include_sensitivities);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86_64"),
        ignore = "from_external_object is unsupported on Intel macOS due to unsupported relocations"
    )]
    #[test]
    fn diffsl_external_object_roundtrip_sparse_f64() {
        type M = crate::FaerSparseMat<f64>;

        let ctx = <M as crate::matrix::MatrixCommon>::C::default();
        let compiled =
            DiffSl::<M, crate::LlvmModule>::compile(serialization_test_model(), ctx, true).unwrap();
        let external_object = compiled.to_external_object().unwrap();
        let rhs_state_deps = external_object.rhs_state_deps.clone();
        let mass_state_deps = external_object.mass_state_deps.clone();
        let include_sensitivities = external_object.include_sensitivities;

        assert!(!rhs_state_deps.is_empty());
        assert!(!mass_state_deps.is_empty());
        assert!(include_sensitivities);

        assert_object_roundtrip::<M>(include_sensitivities);

        let mut imported = DiffSl::<M, ObjectModule>::from_external_object(
            external_object,
            <M as crate::matrix::MatrixCommon>::C::default(),
        )
        .unwrap();
        let p = <M as crate::matrix::MatrixCommon>::C::default().vector_from_vec(vec![3.0, 5.0]);
        imported.set_params(&p);
        assert!(imported.rhs().jacobian_sparsity().is_some());
        assert!(imported.mass().unwrap().sparsity().is_some());
        assert!(imported.rhs_sens_sparsity.is_some());
    }

    #[cfg(feature = "diffsl-llvm")]
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86_64"),
        ignore = "from_external_object is unsupported on Intel macOS due to unsupported relocations"
    )]
    #[test]
    fn diffsl_external_object_roundtrip_dense_f64() {
        type M = crate::NalgebraMat<f64>;

        assert_object_roundtrip::<M>(false);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[cfg_attr(
        all(target_os = "macos", target_arch = "x86_64"),
        ignore = "from_external_object is unsupported on Intel macOS due to unsupported relocations"
    )]
    #[test]
    fn diffsl_serde_roundtrip_object_module_f64() {
        type M = crate::FaerSparseMat<f64>;

        let ctx = <M as crate::matrix::MatrixCommon>::C::default();
        let p = ctx.vector_from_vec(vec![3.0, 5.0]);
        let compiled =
            DiffSl::<M, crate::LlvmModule>::compile(serialization_test_model(), ctx.clone(), true)
                .unwrap();
        let external_object = compiled.to_external_object().unwrap();
        let rhs_state_deps = external_object.rhs_state_deps.clone();
        let rhs_input_deps = external_object.rhs_input_deps.clone();
        let mass_state_deps = external_object.mass_state_deps.clone();

        let mut imported =
            DiffSl::<M, ObjectModule>::from_external_object(external_object, ctx.clone()).unwrap();
        imported.set_params(&p);

        let encoded = serde_json::to_string(&imported).unwrap();
        let mut decoded: DiffSl<M, ObjectModule> = serde_json::from_str(&encoded).unwrap();
        decoded.set_params(&p);

        let t = 0.0;
        let x_imported = imported.init().call(t);
        let x_decoded = decoded.init().call(t);
        x_decoded.assert_eq_st(&x_imported, 1e-10);

        let rhs_imported = imported.rhs().call(&x_imported, t);
        let rhs_decoded = decoded.rhs().call(&x_decoded, t);
        rhs_decoded.assert_eq_st(&rhs_imported, 1e-10);

        assert_eq!(decoded.context.rhs_state_deps, rhs_state_deps);
        assert_eq!(decoded.context.rhs_input_deps, rhs_input_deps);
        assert_eq!(decoded.context.mass_state_deps, mass_state_deps);
        assert!(decoded.rhs().jacobian_sparsity().is_some());
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn diffsl_to_external_object_preserves_deps_after_from_context_f64() {
        type M = crate::FaerSparseMat<f64>;

        let context = DiffSlContext::<M, crate::LlvmModule>::new(
            serialization_test_model(),
            1,
            <M as crate::matrix::MatrixCommon>::C::default(),
        )
        .unwrap();
        let expected_rhs_state_deps = context.rhs_state_deps.clone();
        let expected_rhs_input_deps = context.rhs_input_deps.clone();
        let expected_mass_state_deps = context.mass_state_deps.clone();
        let eqn = DiffSl::from_context(context, true);

        let external_object = eqn.to_external_object().unwrap();
        let rhs_state_deps = external_object.rhs_state_deps;
        let rhs_input_deps = external_object.rhs_input_deps;
        let mass_state_deps = external_object.mass_state_deps;
        let include_sensitivities = external_object.include_sensitivities;

        assert_eq!(rhs_state_deps, expected_rhs_state_deps);
        assert_eq!(rhs_input_deps, expected_rhs_input_deps);
        assert_eq!(mass_state_deps, expected_mass_state_deps);
        assert!(include_sensitivities);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn diffsl_from_external_object_rejects_scalar_type_mismatch() {
        type Mf64 = crate::FaerSparseMat<f64>;
        type Mf32 = crate::FaerSparseMat<f32>;

        let external_object = DiffSl::<Mf64, crate::LlvmModule>::compile(
            serialization_test_model(),
            <Mf64 as crate::matrix::MatrixCommon>::C::default(),
            true,
        )
        .unwrap()
        .to_external_object()
        .unwrap();

        let err = match DiffSl::<Mf32, ObjectModule>::from_external_object(
            external_object,
            <Mf32 as crate::matrix::MatrixCommon>::C::default(),
        ) {
            Ok(_) => panic!("expected scalar type mismatch"),
            Err(err) => err,
        };

        assert!(matches!(err, DiffsolError::Other(_)));
        assert!(err.to_string().contains("scalar type mismatch"));
    }
}
