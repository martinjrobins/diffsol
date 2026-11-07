//! Generate the C header `diffsol.h` of the C API.
//!
//! The header goes to `target/<profile>/include/diffsol.h`, next to the libraries that the build
//! makes. It agrees with the features of the build: it defines a macro (for example
//! `DIFFSOL_DIFFSL_CRANELIFT`) for each enabled feature, and the functions of each feature are
//! inside `#if defined(...)` guards.

use std::{env, fs, path::PathBuf};

/// The features that the header defines a macro for, with the macro names. These must agree
/// with the `[defines]` table in `cbindgen.toml`.
const FEATURES: [(&str, &str); 7] = [
    ("external", "DIFFSOL_EXTERNAL"),
    ("diffsl-external-f64", "DIFFSOL_DIFFSL_EXTERNAL_F64"),
    ("diffsl-external-f32", "DIFFSOL_DIFFSL_EXTERNAL_F32"),
    ("diffsl-external-dynamic", "DIFFSOL_DIFFSL_EXTERNAL_DYNAMIC"),
    ("diffsl-cranelift", "DIFFSOL_DIFFSL_CRANELIFT"),
    ("diffsl-llvm", "DIFFSOL_DIFFSL_LLVM"),
    ("suitesparse", "DIFFSOL_SUITESPARSE"),
];

/// The modules that define C functions with the `c_getter_simple!` and `c_setter_simple!`
/// macros. cbindgen cannot expand macros, so this script writes their declarations itself.
const OPTION_ACCESSOR_FILES: [&str; 2] =
    ["src/ode_options_c.rs", "src/initial_condition_options_c.rs"];

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");
    set_soname();

    let mut config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("cannot read cbindgen.toml");
    let defines: String = FEATURES
        .iter()
        .filter(|(feature, _)| feature_enabled(feature))
        .map(|(_, define)| format!("#define {define} 1\n"))
        .collect();
    config.after_includes = Some(format!(
        "\n/* The features of the library that this header was made with. */\n{defines}"
    ));

    // parse the crate source directly: the C API does not use types of other crates, and
    // `with_crate` would run `cargo metadata` on the whole workspace
    let bindings = cbindgen::Builder::new()
        .with_src(crate_dir.join("src/lib.rs"))
        .with_config(config)
        .generate()
        .expect("cannot generate the C header");
    let mut header = Vec::new();
    bindings.write(&mut header);
    let header = String::from_utf8(header).expect("the C header is not UTF-8");
    let header = insert_option_accessors(header, &crate_dir);

    let path = include_dir().join("diffsol.h");
    write_if_changed(&path, &header);
    // Builds with different features write the same file. If a build with other features
    // changes it, run this script again in the next build with these features, so that the
    // header agrees with the library that this build makes.
    println!("cargo:rerun-if-changed={}", path.display());
}

/// Write `contents` to `path`, but only if it changes the file, so that C builds that depend on
/// the file do not rebuild.
fn write_if_changed(path: &std::path::Path, contents: &str) {
    if fs::read_to_string(path).ok().as_deref() != Some(contents) {
        fs::create_dir_all(path.parent().unwrap()).expect("cannot make the include directory");
        fs::write(path, contents).expect("cannot write the C header");
    }
}

/// On Linux, give the dynamic library the SONAME in the environment variable `DIFFSOL_SONAME`,
/// if it is set. The Debian packaging (`debian/rules`) sets it to `libdiffsol.so.<soversion>`.
///
/// A normal build must not have a SONAME: a program that links a library records its SONAME as
/// the dependency, and Cargo writes only `libdiffsol_c.so`, without a file for the SONAME.
fn set_soname() {
    println!("cargo:rerun-if-env-changed=DIFFSOL_SONAME");
    let Ok(soname) = env::var("DIFFSOL_SONAME") else {
        return;
    };
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,{soname}");
    }
}

fn feature_enabled(feature: &str) -> bool {
    let name = feature.to_uppercase().replace('-', "_");
    env::var_os(format!("CARGO_FEATURE_{name}")).is_some()
}

/// `target/<profile>/include`. `OUT_DIR` is `target/<profile>/build/<package>-<hash>/out`, or
/// `target/<profile>/build/<package>/<hash>/out` with the new build-dir layout of Cargo. Thus use
/// the parent of the nearest `build` directory.
fn include_dir() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    out_dir
        .ancestors()
        .find(|dir| dir.file_name().is_some_and(|name| name == "build"))
        .and_then(|build| build.parent())
        .expect("unexpected OUT_DIR")
        .join("include")
}

/// Add the declarations of the option accessors before the end of the `extern "C"` block.
fn insert_option_accessors(header: String, crate_dir: &std::path::Path) -> String {
    let mut decls = String::from("\n/* The option accessors. */\n");
    for file in OPTION_ACCESSOR_FILES {
        let source = fs::read_to_string(crate_dir.join(file)).expect("cannot read source");
        for (kind, args) in macro_calls(&source) {
            let [prefix, options, value, field] = args;
            let c_value = c_type(&value);
            if kind == "c_getter_simple" {
                decls.push_str(&format!(
                    "\n/* Get the option `{field}`. Return DIFFSOL_OK on success. */\n\
                     int32_t {prefix}_get_{field}(const {options} *options, {c_value} *out_value);\n"
                ));
            } else {
                decls.push_str(&format!(
                    "\n/* Set the option `{field}`. Return DIFFSOL_OK on success. */\n\
                     int32_t {prefix}_set_{field}({options} *options, {c_value} value);\n"
                ));
            }
        }
    }
    let end = header
        .rfind("#ifdef __cplusplus")
        .expect("the header has no end of the extern \"C\" block");
    format!("{}{}\n{}", &header[..end], decls, &header[end..])
}

/// The calls of `c_getter_simple!` and `c_setter_simple!` in `source`, with their four
/// arguments.
fn macro_calls(source: &str) -> Vec<(&'static str, [String; 4])> {
    let mut calls = Vec::new();
    for kind in ["c_getter_simple", "c_setter_simple"] {
        let start = format!("{kind}!(");
        let mut rest = source;
        while let Some(i) = rest.find(&start) {
            rest = &rest[i + start.len()..];
            let end = rest.find(')').expect("unterminated macro call");
            let args: Vec<String> = rest[..end]
                .split(',')
                .map(|a| a.trim().to_string())
                .filter(|a| !a.is_empty())
                .collect();
            let args: [String; 4] = args.try_into().expect("expected four macro arguments");
            calls.push((kind, args));
            rest = &rest[end..];
        }
    }
    // keep each getter next to its setter
    calls.sort_by(|a, b| (&a.1[0], &a.1[3], a.0).cmp(&(&b.1[0], &b.1[3], b.0)));
    calls
}

fn c_type(rust_type: &str) -> &'static str {
    match rust_type {
        "usize" => "size_t",
        "i32" => "int32_t",
        "f64" => "double",
        other => panic!("no C type for the option type `{other}`"),
    }
}
