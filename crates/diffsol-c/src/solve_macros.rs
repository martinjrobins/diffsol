// macro to generate all the trait methods for accessing ic_options
#[macro_export]
macro_rules! generate_trait_ic_option_accessors {
    ($($field:ident : $type:ty),*) => {
        $(
            paste! {
                fn [<set_ic_ $field>](&mut self, value: $type);
                fn [<ic_ $field>](&self) -> $type;
            }
        )*
    };
}

// macro to generate all the trait methods for accessing ode_options
#[macro_export]
macro_rules! generate_trait_ode_option_accessors {
    ($($field:ident : $type:ty),*) => {
        $(
            paste! {
                fn [<set_ode_ $field>](&mut self, value: $type);
                fn [<ode_ $field>](&self) -> $type;
            }
        )*
    };
}

// helper macros to convert f64 to/from M::T when needed
#[macro_export]
macro_rules! option_value_to_store {
    ($value:expr, f64) => {
        M::T::from_f64($value).unwrap()
    };
    ($value:expr, $type:ty) => {
        $value
    };
}

#[macro_export]
macro_rules! option_value_from_store {
    ($value:expr, f64) => {
        $value.to_f64().unwrap()
    };
    ($value:expr, $type:ty) => {
        $value
    };
}

// generic accessor generator to reduce duplication between ic/ode variants
#[macro_export]
macro_rules! generate_option_accessors {
    ($store:ident, $prefix:ident; $($field:ident : $type:ty),* $(,)?) => {
        $(
            paste! {
                fn [<set_ $prefix _ $field>](&mut self, value: $type) {
                    self.problem.$store.$field = option_value_to_store!(value, $type);
                }

                fn [<$prefix _ $field>](&self) -> $type {
                    option_value_from_store!(self.problem.$store.$field, $type)
                }
            }
        )*
    };
}

// macro to generate all the setters and getters for ic_options
#[macro_export]
macro_rules! generate_ic_option_accessors {
    ($($field:ident : $type:ty),* $(,)?) => {
        generate_option_accessors! { ic_options, ic; $($field : $type),* }
    };
}

// macro to generate all the setters and getters for ode_options
#[macro_export]
macro_rules! generate_ode_option_accessors {
    ($($field:ident : $type:ty),* $(,)?) => {
        generate_option_accessors! { ode_options, ode; $($field : $type),* }
    };
}
