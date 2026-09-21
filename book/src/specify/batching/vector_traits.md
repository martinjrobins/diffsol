# Vector Traits

While diffsol aims to hide the specifics of batching from the user as much as possible, it is sometimes neccessary to work with batched vectors directly. The `Vector` trait provides a set of methods for working with batched vectors, including methods for accessing the number of batch lanes, the inner representation of the vector, and methods for iterating over batch lanes and elements.

For the trivial case of a single batch lane for a vector living on the host, the standard `Index`  and `IndexMut` traits are implemented and can be used to access the elements of the vector. Note that these will panic if the vector has more than one batch lane, so they should only be used for single-batch vectors.

## A batched Vector

To create a batched vector with `NBATCH = 3` batches, you can create a `Context` with the number of batches and create a new vector with that specific context. The `Vector::from_vec` method takes a vector of length `NBATCH * N` and splits it into `NBATCH` separate vectors of length `N`.

```rust,ignore
{{#include ../../../../examples/batching/src/main.rs:vector_create}}
```

All the backends store the batches as columns in a matrix type defined by that backend. For example we are using the `nalgebra` backend, which stores the vector using a `DMatrix` type, which you can obtain by using the `Vector::inner` method.

```rust,ignore
{{#include ../../../../examples/batching/src/main.rs:vector_inner}}
```

## "for each" operations

Typically when you are using a vector you don't want to have to worry about the specifics of the number of batches. This is particularly the case if you are working with more than one vector with different numbers of batches, where broadcasting rules become important.

To mitigate this, the `Vector` trait provies a set of methods for working with individual batches/elements without worrying about batching. The `Vector::for_each_batch` method allows you to provide a closure that is executed for each batch lane, while vectors with a lesser (but compatible) number of batches will be broadcast to match the output vector, which must have the largest number of batches.

```rust,ignore
{{#include ../../../../examples/batching/src/main.rs:vector_for_each}}
```

So that it can be executed on a GPU device (if applicable), the closure passed to `for_each_batch` must be of type `Fn` and be both `Copy` and `Send`. If this is too restrictive and you don't care about solving on the GPU, you can instead use the `Vector::for_each_batch_host` method, which allows you to pass a closure of type `FnMut` that is not required to be either `Copy` or `Send`.

```rust,ignore
{{#include ../../../../examples/batching/src/main.rs:vector_for_each_host}}
```

Unless your ODE system is very small, when executing on the GPU you typically want to use one thread for each state vector element, rather than one thread for each batch lane. The `Vector::for_each_elem` method allows you to provide a closure that is executed for each element of the vector.

```rust,ignore
{{#include ../../../../examples/batching/src/main.rs:vector_for_each_elem}}
```

## Reductions

The `Vector` trait also has a couple of generic reduction methods for vectors. The `Vector::reduce_elem` method allows you to reduce over the elements of a vector for each batch lane, giving a final vector with `N = 1` elements and `NBATCH` batch lanes.

```rust,ignore
{{#include ../../../../examples/batching/src/main.rs:vector_reduce_elem}}
```

The `Vector::reduce_batch` method allows you to reduce over the batch lanes of a vector, giving a final vector with `N` elements and `NBATCH = 1` batch lanes.

```rust,ignore
{{#include ../../../../examples/batching/src/main.rs:vector_reduce_batch}}
```
