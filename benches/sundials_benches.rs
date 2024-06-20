use std::ffi::c_int;

extern "C" {
    pub fn idaFoodWeb_bnd_5() -> c_int;
    pub fn idaFoodWeb_bnd_10() -> c_int;
    pub fn idaFoodWeb_bnd_20() -> c_int;
    pub fn idaHeat2d_klu_5() -> c_int;
    pub fn idaHeat2d_klu_10() -> c_int;
    pub fn idaHeat2d_klu_20() -> c_int;
    pub fn idaRoberts_dns() -> c_int;
}
