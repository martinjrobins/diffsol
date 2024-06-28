use cc::Build;
use std::env;

fn compile_benches() -> Vec<String> {
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

    let includes = [
        env::var("DEP_SUNDIALS_KLU_INCLUDE").unwrap().to_string(),
        format!("{}/include", env::var("DEP_SUNDIALS_ROOT").unwrap()),
    ];
    let libname = "sundials_benches";
    Build::new()
        .files(files.clone())
        .includes(includes)
        .compile(libname);
    files.into_iter().map(|s| s.to_string()).collect()
}

fn main() {
    // return if we are not using the sundials and suitesparse features
    if !cfg!(feature = "sundials") {
        return;
    }

    // compile sundials benches
    let mut files = compile_benches();
    files.push("benches/sundials_benches.rs".to_string());
    files.push("build.rs".to_string());
    files.push("benches/idaHeat2d_klu.c".to_string());
    files.push("benches/idaFoodWeb_bnd.c".to_string());

    for name in files.into_iter() {
        println!("cargo:rerun-if-changed={}", name);
    }
}
