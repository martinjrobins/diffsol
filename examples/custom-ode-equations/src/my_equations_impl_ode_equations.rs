use crate::{MyEquations, MyInit, MyMass, MyOut, MyRhs, MyRoot, V};
use diffsol::{OdeEquations, OdeEquationsRef, Vector};

impl<'a> OdeEquationsRef<'a> for MyEquations {
    type Rhs = MyRhs<'a>;
    type Mass = MyMass<'a>;
    type Init = MyInit<'a>;
    type Root = MyRoot<'a>;
    type Out = MyOut<'a>;
}

impl OdeEquations for MyEquations {
    fn rhs(&self) -> <MyEquations as OdeEquationsRef<'_>>::Rhs {
        MyRhs { p: &self.p }
    }
    fn mass(&self) -> Option<<MyEquations as OdeEquationsRef<'_>>::Mass> {
        Some(MyMass { p: &self.p })
    }
    fn init(&self) -> <MyEquations as OdeEquationsRef<'_>>::Init {
        MyInit { p: &self.p }
    }
    fn root(&self) -> Option<<MyEquations as OdeEquationsRef<'_>>::Root> {
        Some(MyRoot { p: &self.p })
    }
    fn out(&self) -> Option<<MyEquations as OdeEquationsRef<'_>>::Out> {
        Some(MyOut { p: &self.p })
    }
    fn set_params(&mut self, p: &V) {
        self.p.copy_from(p);
    }
    fn get_params(&self, p: &mut Self::V) {
        p.copy_from(&self.p);
    }
}
