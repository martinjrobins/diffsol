use cc::Build;
use std::env;

fn compile_benches() -> Vec<String> {
    let files = [
        "benches/idaFoodWeb_bnd_5.c",
        "benches/idaFoodWeb_bnd_10.c",
        "benches/idaFoodWeb_bnd_20.c",
        "benches/idaHeat2d_bnd_5.c",
        "benches/idaHeat2d_bnd_10.c",
        "benches/idaHeat2d_bnd_20.c",
        "benches/idaRoberts_dns.c",
    ];
    let includes = [
        env::var("DEP_SUNDIALS_KLU_INCLUDE").unwrap(),
        format!("{}/include", env::var("DEP_SUNDIALS_ROOT").unwrap()),
    ];
    let libname = "sundials_benches";
    Build::new()
        .files(files)
        .includes(includes)
        .compile(libname);
    files.into_iter().map(|s| s.to_string()).collect()
}

fn main() {
    // return if we are not using the sundials feature
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
