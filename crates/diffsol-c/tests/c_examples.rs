//! Compile and run the C examples in `examples/c` against the dynamic library and the header
//! that this build made. Each example returns a non-zero exit code if its solution is not
//! correct.
//!
//! The `cc` crate finds the C compiler: `$CC` or the default compiler of the platform, also MSVC
//! on Windows.

use std::path::{Path, PathBuf};
use std::process::Command;

/// `target/<profile>/deps`, which holds the dynamic library of this build. The test executable is
/// in the same directory. (`cargo test` does not copy the library to `target/<profile>`, as
/// `cargo build` does.)
fn library_dir() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    exe.parent().unwrap().to_path_buf()
}

/// Compile the example `example`, and run it with the environment variables `envs`.
fn compile_and_run(example: &str, envs: &[(&str, &str)]) {
    let library = library_dir();
    // the header that the build script of this build wrote
    let include = PathBuf::from(env!("DIFFSOL_C_INCLUDE_DIR"));
    assert!(
        include.join("diffsol.h").exists(),
        "the build did not make {}",
        include.join("diffsol.h").display()
    );
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("c")
        .join(format!("{example}.c"));
    let out_dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let exe = out_dir.join(format!("{example}{}", std::env::consts::EXE_SUFFIX));

    let compiler = cc::Build::new()
        .target(env!("DIFFSOL_C_TARGET"))
        .host(env!("DIFFSOL_C_HOST"))
        .opt_level(0)
        .cargo_metadata(false)
        .cargo_warnings(false)
        .get_compiler();
    let mut command = compiler.to_command();
    // the compiler writes its intermediate files to the current directory
    command.current_dir(out_dir).arg(&source);
    if compiler.is_like_msvc() {
        command
            .args(["/nologo", "/W3"])
            .arg(format!("/I{}", include.display()))
            .arg(format!("/Fe{}", exe.display()))
            .arg(library.join("diffsol_c.dll.lib"));
    } else {
        command
            .args(["-std=c11", "-Wall", "-Wextra", "-Werror"])
            .arg(format!("-I{}", include.display()))
            .arg(format!("-L{}", library.display()))
            .arg("-ldiffsol_c")
            .arg("-o")
            .arg(&exe);
        if !cfg!(windows) {
            command
                .arg("-lm")
                .arg(format!("-Wl,-rpath,{}", library.display()));
        }
    }
    let output = command
        .output()
        .unwrap_or_else(|err| panic!("cannot run the C compiler {:?}: {err}", compiler.path()));
    assert!(
        output.status.success(),
        "cannot compile {example}.c:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let mut run = Command::new(&exe);
    run.envs(envs.iter().copied());
    if cfg!(windows) {
        // Windows finds the DLL with PATH
        let path = std::env::var_os("PATH").unwrap_or_default();
        let mut paths = vec![library.clone()];
        paths.extend(std::env::split_paths(&path));
        run.env("PATH", std::env::join_paths(paths).unwrap());
    }
    let output = run.output().unwrap();
    assert!(
        output.status.success(),
        "{example} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn c_example_logistic_callbacks() {
    compile_and_run("logistic_callbacks", &[]);
}

#[cfg(any(feature = "diffsl-cranelift", feature = "diffsl-llvm"))]
#[test]
fn c_example_logistic_diffsl_jit() {
    compile_and_run("logistic_diffsl_jit", &[]);
}

#[cfg(feature = "diffsl-llvm")]
#[test]
fn c_example_logistic_diffsl_jit_llvm() {
    compile_and_run("logistic_diffsl_jit", &[("DIFFSOL_JIT_BACKEND", "llvm")]);
}
