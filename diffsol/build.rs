#[cfg(feature = "sundials")]
mod if_sundials {

    use cc::Build;
    use std::io::Read;
    use std::{collections::HashSet, env, fs::File, io::BufReader, path::PathBuf};

    pub const LINK_SUNDIALS_LIBRARIES: &[&str] = &[
        "cvode",
        "ida",
        "sunlinsolklu",
        "nvecserial",
        "sunlinsolband",
        "sunlinsoldense",
        "sunlinsolklu",
        "sunmatrixband",
        "sunmatrixdense",
        "sunmatrixsparse",
        "sunnonlinsolfixedpoint",
        "sunnonlinsolnewton",
    ];

    // SUNDIALS has a few non-negative constants that need to be parsed as an i32.
    // This is an attempt at doing so generally.
    #[derive(Debug)]
    struct ParseSignedConstants;

    impl bindgen::callbacks::ParseCallbacks for ParseSignedConstants {
        fn int_macro(&self, name: &str, _value: i64) -> Option<bindgen::callbacks::IntKind> {
            let prefix: String = name.chars().take_while(|c| *c != '_').collect();
            match prefix.as_ref() {
                "CV" | "IDA" | "KIN" | "SUN" => Some(bindgen::callbacks::IntKind::Int),
                _ => None,
            }
        }
    }

    // Ignore some macros (based on https://github.com/rust-lang/rust-bindgen/issues/687#issuecomment-1312298570)
    #[derive(Debug)]
    struct IgnoreMacros(HashSet<&'static str>);

    impl bindgen::callbacks::ParseCallbacks for IgnoreMacros {
        fn will_parse_macro(&self, name: &str) -> bindgen::callbacks::MacroParsingBehavior {
            use bindgen::callbacks::MacroParsingBehavior;
            if self.0.contains(name) {
                MacroParsingBehavior::Ignore
            } else {
                MacroParsingBehavior::Default
            }
        }
    }

    impl IgnoreMacros {
        const IGNORE_CONSTANTS: [&'static str; 19] = [
            "FE_DIVBYZERO",
            "FE_DOWNWARD",
            "FE_INEXACT",
            "FE_INVALID",
            "FE_OVERFLOW",
            "FE_TONEAREST",
            "FE_TOWARDZERO",
            "FE_UNDERFLOW",
            "FE_UPWARD",
            "FP_INFINITE",
            "FP_INT_DOWNWARD",
            "FP_INT_TONEAREST",
            "FP_INT_TONEARESTFROMZERO",
            "FP_INT_TOWARDZERO",
            "FP_INT_UPWARD",
            "FP_NAN",
            "FP_NORMAL",
            "FP_SUBNORMAL",
            "FP_ZERO",
        ];

        fn new() -> Self {
            Self(Self::IGNORE_CONSTANTS.iter().copied().collect())
        }
    }

    #[derive(Debug)]
    pub struct Library {
        pub inc: Option<String>,
        pub lib: Option<String>,
    }

    impl Library {
        pub fn new_sundials() -> Self {
            Self {
                inc: env::var("SUNDIALS_INCLUDE_DIR").ok(),
                lib: env::var("SUNDIALS_LIBRARY_DIR").ok(),
            }
        }
        pub fn new_suitesparse() -> Self {
            Self {
                inc: env::var("DEP_SUITESPARSE_SUITESPARSE_INCLUDE").ok(),
                lib: env::var("DEP_SUITESPARSE_SUITESPARSE_LIB").ok(),
            }
        }
    }

    pub fn compile_benches(sundials: &Library, suitesparse: &Library) -> Vec<String> {
        let basedir = "benches/sundials/";
        let files = [
            "cvRoberts_block_klu.c",
            "idaFoodWeb_bnd_5.c",
            "idaFoodWeb_bnd_10.c",
            "idaFoodWeb_bnd_20.c",
            "idaFoodWeb_bnd_30.c",
            "idaHeat2d_bnd_5.c",
            "idaHeat2d_bnd_10.c",
            "idaHeat2d_bnd_20.c",
            "idaHeat2d_bnd_30.c",
            "idaRoberts_dns.c",
        ];
        let files = files
            .iter()
            .map(|f| format!("{}{}", basedir, f))
            .collect::<Vec<_>>();

        let mut includes = vec![];
        if let Some(dir) = suitesparse.inc.as_ref() {
            includes.push(dir.to_string());
        }
        if let Some(dir) = sundials.inc.as_ref() {
            includes.push(dir.to_string());
        }
        let libname = "sundials_benches";
        Build::new()
            .files(files.clone())
            .includes(includes)
            .flag("-Wno-unused-function")
            .compile(libname);
        files.into_iter().map(|s| s.to_string()).collect()
    }

    pub fn generate_bindings(suitesparse: &Library, sundials: &Library) -> Result<PathBuf, String> {
        let mut builder = bindgen::Builder::default().header("wrapper.h");
        if let Some(dir) = sundials.inc.as_ref() {
            builder = builder.clang_arg(format!("-I{}", dir));
        }
        if let Some(dir) = suitesparse.inc.as_ref() {
            builder = builder.clang_arg(format!("-I{}", dir));
        }
        let bindings = builder
            .parse_callbacks(Box::new(IgnoreMacros::new()))
            .parse_callbacks(Box::new(ParseSignedConstants))
            .generate()
            .map_err(|e| e.to_string())?;

        let bindings_rs = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");

        bindings
            .write_to_file(bindings_rs.clone())
            .expect("Couldn't write file bindings.rs!");
        Ok(bindings_rs)
    }

    pub fn get_sundials_version_major(bindings: PathBuf) -> Option<u32> {
        let b = File::open(bindings).expect("Couldn't read file bindings.rs!");
        let mut b = BufReader::new(b).bytes();
        'version: while b.any(|c| c.as_ref().is_ok_and(|&c| c == b'S')) {
            for c0 in "UNDIALS_VERSION_MAJOR".bytes() {
                match b.next() {
                    Some(Ok(c)) => {
                        if c != c0 {
                            continue 'version;
                        }
                    }
                    Some(Err(_)) | None => return None,
                }
            }
            // Match " : u32 = 6"
            if b.any(|c| c.as_ref().is_ok_and(|&c| c == b'=')) {
                let is_not_digit = |c: &u8| !c.is_ascii_digit();
                let b = b.skip_while(|c| c.as_ref().is_ok_and(is_not_digit));
                let v: Vec<_> = b
                    .map_while(|c| c.ok().filter(|c| c.is_ascii_digit()))
                    .collect();
                match String::from_utf8(v) {
                    Ok(v) => return v.parse().ok(),
                    Err(_) => return None,
                }
            }
            return None;
        }
        None
    }
}

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

    #[cfg(feature = "sundials")]
    {
        use if_sundials::*;
        // return if we are not using the sundials features
        if !cfg!(feature = "sundials") {
            return Ok(());
        }
        let sundials = Library::new_sundials();
        let suitesparse = Library::new_suitesparse();

        // generate sundials bindings
        let bindings_rs = generate_bindings(&suitesparse, &sundials)?;
        let major = get_sundials_version_major(bindings_rs).unwrap_or(5);
        println!("cargo:rustc-cfg=sundials_version_major=\"{}\"", major);
        println!("cargo::rustc-check-cfg=cfg(sundials_version_major, values(\"5\", \"6\", \"7\"))");

        // compile sundials benches
        let mut files = compile_benches(&sundials, &suitesparse);
        files.push("benches/sundials_benches.rs".to_string());
        files.push("benches/idaHeat2d_klu_v5.c".to_string());
        files.push("benches/idaHeat2d_klu_v6.c".to_string());
        files.push("benches/idaFoodWeb_bnd_v5.c".to_string());
        files.push("benches/idaFoodWeb_bnd_v6.c".to_string());
        for name in files.into_iter() {
            println!("cargo:rerun-if-changed={}", name);
        }

        // link to sundials
        if let Some(dir) = sundials.lib.as_ref() {
            println!("cargo:rustc-link-search=native={}", dir);
        }
        let library_type = "dylib";
        for lib_name in LINK_SUNDIALS_LIBRARIES {
            println!(
                "cargo:rustc-link-lib={}=sundials_{}",
                library_type, lib_name
            );
        }
        // link to klu
        if let Some(dir) = suitesparse.lib.as_ref() {
            println!("cargo:rustc-link-search=native={}", dir);
            println!("cargo:rustc-env=LD_LIBRARY_PATH={}", dir);
        }
        println!("cargo:rustc-link-lib={}=klu", library_type);
    }
    Ok(())
}
