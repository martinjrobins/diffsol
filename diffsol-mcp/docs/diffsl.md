# DiffSL DSL Reference

This document is a compact reference for the DiffSL language used by the `diffsol-mcp` server.
It combines the canonical parser grammar from the DiffSL repository with the user-facing syntax
documentation from the DiffSL book under `/home/mrobins/git/diffsl/book/src`.

## Purpose

DiffSL is a DSL for defining ODE and DAE systems in the form:

$$
M(t) \frac{\mathrm{d}\mathbf{u}}{\mathrm{d}t} = F(\mathbf{u}, t)
$$

where:

- `u_i` is the state vector and its values are the initial conditions
- `F_i` is the right-hand side of the equations
- `dudt_i` is optional and provides initial derivative guesses
- `M_i` is optional and represents the left-hand side `M * dudt`
- additional scalars, vectors, and tensors can be defined as intermediates
- `in` provides named runtime parameters with default values
- `out` or `out_i` defines outputs

## Minimal Example

```diffsl
in_i { r = 1.0, k = 10.0 }
u_i {
  N = 0.0
}
F_i {
  r * N * (1 - N / k)
}
out_i {
  N
}
```

## Core Tensors

### Scalars

```diffsl
k { 1.0 }
```

### Vectors

```diffsl
v_i { 1.0, 2.0, 3.0 }
```

### Matrices

```diffsl
A_ij {
  (0:2, 0:3): 1.0,
}
```

The number of subscripts determines tensor rank:

- `k` is scalar
- `v_i` is 1D
- `A_ij` is 2D

Index labels are semantic. They matter for broadcasting, permutation, contraction, and indexing.

## Named Elements

Tensor elements can be labelled and then referenced by name:

```diffsl
u_i {
  x = 1.0,
  y = 2.0,
}
w_i {
  2 * y,
  3 * x,
}
```

## Inputs and Outputs

Inputs define named runtime parameters with defaults:

```diffsl
in { k = 1.0 }
u { 0.1 }
F { k * u }
```

Outputs can expose state variables or derived values:

```diffsl
u_i {
  x = 1.0,
  y = 2.0,
  z = 3.0,
}
out_i { x, y, z }
```

or:

```diffsl
u_i {
  x = 1.0,
  y = 2.0,
  z = 3.0,
}
out { x + y + z }
```

## Defining ODEs

The required tensor for an ODE is `F_i`.

```diffsl
u_i { x = 1, y = 0 }
F_i { y, -x }
```

For DAEs or non-identity mass matrices, define `dudt_i` and `M_i` as well:

```diffsl
u_i {
  x = 1,
  y = 0,
}
dudt_i {
  dxdt = 0,
  dydt = 1,
}
M_i {
  dxdt,
  0,
}
F_i {
  x,
  y - x,
}
```

`M_i` is the product `M * dudt`, not the mass matrix entries themselves.

## Expressions and Operations

Supported arithmetic:

- `+`
- `-`
- `*`
- `/`

Supported built-in functions:

- `pow(x, y)`
- `sin(x)`
- `cos(x)`
- `tan(x)`
- `exp(x)`
- `log(x)`
- `sqrt(x)`
- `abs(x)`
- `sigmoid(x)`
- `heaviside(x)`

Built-in variable:

- `t` for current time

Example:

```diffsl
F_i {
  t + sin(t)
}
```

## Broadcasting and Contraction

Broadcasting is index-based rather than purely shape-based.

Valid:

```diffsl
A_ij { (0:3, 0:2): 1.0 }
b_i { (0:2): 1.0 }
c_ij { A_ij + b_j }
```

Invalid:

```diffsl
A_ij { (0:3, 0:2): 1.0 }
b_i { (0:2): 1.0 }
c_ij { A_ij + b_i }
```

Contractions sum over indices that do not appear in the output tensor:

```diffsl
v_i { A_ij * u_j }
```

## Indexing

You can index dense 1D tensors with square brackets:

```diffsl
a_i { 0.0, 1.0, 2.0, 3.0 }
r { a_i[2] }
```

Ranges are also supported:

```diffsl
a_i { 0.0, 1.0, 2.0, 3.0 }
r_i { a_i[1:3] }
```

## Sparse and Diagonal Structure

The compiler infers sparse or diagonal structure from the specified non-zero elements.

Sparse:

```diffsl
B_ij {
  (0, 0): 1.0,
  (0, 1): 2.0,
  (1, 1): 3.0,
}
```

Diagonal:

```diffsl
D_ij {
  (0, 0): 1.0,
  (1, 1): 2.0,
  (2, 2): 3.0,
}
```

Diagonal ranges can also be written with `..`:

```diffsl
D_ij {
  (0..2, 0..2): 1.0,
}
```

## Parser Grammar

The canonical parser grammar is currently:

```pest
main       = { SOI ~ model ~ EOI }
model      = { inputs? ~ tensor* }
inputs     = { "in" ~ "=" ~ "[" ~ name? ~ (DELIM ~ name)* ~ DELIM? ~ "]" }
tensor     = { name_ij ~ "{" ~ tensor_elmt? ~ (DELIM ~ tensor_elmt)* ~ DELIM? ~ "}" }
indices   = { "(" ~ indice ~ ("," ~ indice)* ~ ")" ~ ":" }
indice      = { integer ~ ( range_sep ~ integer )? }
tensor_elmt = { indices? ~ (assignment | expression) }
assignment = { name ~ "=" ~ expression }
expression = { term ~ (term_op ~ term)* }
term       = { factor ~ (factor_op ~ factor)* }
factor     = { sign? ~ ( call | real | integer | name_ij_index | name_ij | "(" ~ expression ~ ")" ) }
integer_expression = { integer_term ~ (term_op ~ integer_term)* }
integer_term       = { integer_factor ~ (integer_factor_op ~ integer_factor)* }
integer_factor     = { sign? ~ ( integer | integer_name | "(" ~ integer_expression ~ ")" ) }
call       = { name ~ "(" ~ call_arg ~ ("," ~ call_arg )* ~ ")" }
call_arg   = { expression }
name_ij    = ${ name ~ ("_" ~ name)? }
index_indice = { integer_expression ~ ( range_sep ~ integer_expression )? }
name_ij_index = { name_ij ~ "[" ~ index_indice ~ "]" }
range_sep  = @{ ".." | ":" }
sign       = @{ ("-"|"+") }
term_op    = @{ "-"|"+" }
factor_op  = @{ "*"|"/" }
integer_factor_op  = @{ "*"|"/"|"%" }
integer_name = @{ "N" }
name       = @{ ( 'a'..'z' | 'A'..'Z' ) ~ ('a'..'z' | 'A'..'Z' | '0'..'9' )* }
integer    = @{ ('0'..'9')+ }
real       = @{ ('0'..'9')+ ~ ( "." ~ ('0'..'9')* )? ~ ( "e" ~ sign? ~ integer )? }
DELIM      = _{ "," }
WHITESPACE = _{ " " | NEWLINE | "\t" }
COMMENT    = _{
    "/*" ~ (!"*/" ~ ANY)* ~ "*/"
    | "//" ~ (!NEWLINE ~ ANY)*
}
```

## Source Material

This file was assembled from:

- `https://raw.githubusercontent.com/martinjrobins/diffsl/refs/heads/main/diffsl/src/parser/ds_grammar.pest`
- `https://raw.githubusercontent.com/martinjrobins/diffsl/refs/heads/main/book/src/introduction.md`
- `https://raw.githubusercontent.com/martinjrobins/diffsl/refs/heads/main/book/src/inputs_outputs.md`
- `https://raw.githubusercontent.com/martinjrobins/diffsl/refs/heads/main/book/src/tensors.md`
- `https://raw.githubusercontent.com/martinjrobins/diffsl/refs/heads/main/book/src/functions.md`
- `https://raw.githubusercontent.com/martinjrobins/diffsl/refs/heads/main/book/src/operations.md`
- `https://raw.githubusercontent.com/martinjrobins/diffsl/refs/heads/main/book/src/odes.md`
