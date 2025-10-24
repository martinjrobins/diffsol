use diffsl::execution::module::CodegenModule;
use std::collections::HashMap;

pub type RealType = f64;
pub type UIntType = u32;

extern "C" {
    pub fn set_constants(thread_id: UIntType, thread_dim: UIntType);
    pub fn stop(
        time: RealType,
        u: *const RealType,
        data: *mut RealType,
        root: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    pub fn rhs(
        time: RealType,
        u: *const RealType,
        data: *mut RealType,
        rr: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    pub fn rhs_grad(
        time: RealType,
        u: *const RealType,
        du: *const RealType,
        data: *const RealType,
        ddata: *mut RealType,
        rr: *const RealType,
        drr: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );

    pub fn mass(
        time: RealType,
        u: *const RealType,
        data: *mut RealType,
        mv: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    pub fn set_u0(u: *mut RealType, data: *mut RealType, thread_id: UIntType, thread_dim: UIntType);

    pub fn set_u0_grad(
        u: *const RealType,
        du: *mut RealType,
        data: *const RealType,
        ddata: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );

    pub fn calc_out(
        time: RealType,
        u: *const RealType,
        data: *mut RealType,
        out: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    pub fn calc_out_grad(
        time: RealType,
        u: *const RealType,
        du: *const RealType,
        data: *const RealType,
        ddata: *mut RealType,
        out: *const RealType,
        dout: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );

    pub fn get_dims(
        states: *mut UIntType,
        inputs: *mut UIntType,
        outputs: *mut UIntType,
        data: *mut UIntType,
        stop: *mut UIntType,
        has_mass: *mut UIntType,
    );
    pub fn set_inputs(inputs: *const RealType, data: *mut RealType);
    pub fn get_inputs(inputs: *mut RealType, data: *const RealType);
    pub fn set_inputs_grad(
        inputs: *const RealType,
        dinputs: *const RealType,
        data: *const RealType,
        ddata: *mut RealType,
    );

    pub fn set_id(id: *mut RealType);
    pub fn get_tensor(
        data: *const RealType,
        tensor_data: *mut *mut RealType,
        tensor_size: *mut UIntType,
    );
}

#[cfg(feature = "diffsl-ext-sens")]
extern "C" {
    pub fn rhs_rgrad(
        time: RealType,
        u: *const RealType,
        du: *mut RealType,
        data: *const RealType,
        ddata: *mut RealType,
        rr: *const RealType,
        drr: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    pub fn set_u0_rgrad(
        u: *const RealType,
        du: *mut RealType,
        data: *const RealType,
        ddata: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    pub fn calc_out_rgrad(
        time: RealType,
        u: *const RealType,
        du: *mut RealType,
        data: *const RealType,
        ddata: *mut RealType,
        out: *const RealType,
        dout: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    pub fn set_inputs_rgrad(
        inputs: *const RealType,
        dinputs: *mut RealType,
        data: *const RealType,
        ddata: *mut RealType,
    );

    pub fn rhs_srgrad(
        time: RealType,
        u: *const RealType,
        data: *const RealType,
        ddata: *mut RealType,
        rr: *const RealType,
        drr: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    pub fn calc_out_srgrad(
        time: RealType,
        u: *const RealType,
        data: *const RealType,
        ddata: *mut RealType,
        out: *const RealType,
        dout: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
}

#[cfg(feature = "diffsl-ext-sens")]
extern "C" {
    pub fn rhs_sgrad(
        time: RealType,
        u: *const RealType,
        data: *const RealType,
        ddata: *mut RealType,
        rr: *const RealType,
        drr: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    pub fn set_u0_sgrad(
        u: *const RealType,
        du: *mut RealType,
        data: *const RealType,
        ddata: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
    pub fn calc_out_sgrad(
        time: RealType,
        u: *const RealType,
        data: *const RealType,
        ddata: *mut RealType,
        out: *const RealType,
        dout: *mut RealType,
        thread_id: UIntType,
        thread_dim: UIntType,
    );
}

pub struct ExtLinkModule;

impl CodegenModule for ExtLinkModule {}

pub(crate) fn symbol_map() -> HashMap<String, *const u8> {
    let mut map = HashMap::new();
    map.insert("set_u0".to_string(), set_u0 as *const u8);
    map.insert("rhs".to_string(), rhs as *const u8);
    map.insert("mass".to_string(), mass as *const u8);
    map.insert("calc_out".to_string(), calc_out as *const u8);
    map.insert("calc_stop".to_string(), stop as *const u8);
    map.insert("set_id".to_string(), set_id as *const u8);
    map.insert("get_dims".to_string(), get_dims as *const u8);
    map.insert("set_inputs".to_string(), set_inputs as *const u8);
    map.insert("get_inputs".to_string(), get_inputs as *const u8);
    map.insert("set_constants".to_string(), set_constants as *const u8);

    map.insert("set_u0_grad".to_string(), set_u0_grad as *const u8);
    map.insert("rhs_grad".to_string(), rhs_grad as *const u8);
    map.insert("calc_out_grad".to_string(), calc_out_grad as *const u8);
    map.insert("set_inputs_grad".to_string(), set_inputs_grad as *const u8);

    #[cfg(feature = "diffsl-ext-sens")]
    {
        map.insert("set_u0_rgrad".to_string(), set_u0_rgrad as *const u8);
        map.insert("rhs_rgrad".to_string(), rhs_rgrad as *const u8);
        map.insert("calc_out_rgrad".to_string(), calc_out_rgrad as *const u8);
        map.insert(
            "set_inputs_rgrad".to_string(),
            set_inputs_rgrad as *const u8,
        );

        map.insert("rhs_srgrad".to_string(), rhs_srgrad as *const u8);
        map.insert("calc_out_srgrad".to_string(), calc_out_srgrad as *const u8);
    }

    #[cfg(feature = "diffsl-ext-sens")]
    {
        map.insert("rhs_sgrad".to_string(), rhs_sgrad as *const u8);
        map.insert("calc_out_sgrad".to_string(), calc_out_sgrad as *const u8);
        map.insert("set_u0_sgrad".to_string(), set_u0_sgrad as *const u8);
    }

    map
}
