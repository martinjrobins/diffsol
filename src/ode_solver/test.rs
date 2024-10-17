type V = Vec<i8>;

struct Data {
    y: V,
    g: V
}

struct DataThingWrapperMut<'a> {
    y: &'a mut V,
    g: &'a mut V
}

trait DataThing {
    fn get_mut(&mut self) -> DataThingWrapperMut;
}

impl DataThing for Data {
    fn get_mut(&mut self) -> DataThingWrapperMut {
        DataThingWrapperMut {
            y: &mut self.y,
            g: &mut self.g
        }
    }
}

fn test1(a: &V) {
}

fn test2(b: &mut V) {
}

fn test3(a: &V, b: &mut V) {
}

fn make_data() -> Data {
    Data{
        y: vec![1, 2, 3],
        g: vec![]
    }
}

fn main() {
    {
        let d1 = make_data();
        test1(&d1.get_mut().y);
    }

    {
        let mut d2 = make_data();
        test2(&mut d2.get_mut().g);
    }

    {
        let mut d3 = make_data();
        let mut d3_mut = d3.get_mut();
        test3(&d3_mut.y, &mut d3_mut.g);
    }
}