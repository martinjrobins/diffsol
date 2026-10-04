# a python script that extracts the rust code blocks from README.md
# and writes them to the lorenz-attractor-diffsl-llvm example, in order.
# This is used in a github workflow to keep the example code in sync
import re

OUTPUTS = [
    "examples/lorenz-attractor-diffsl-llvm/src/lorenz_closure.rs",
    "examples/lorenz-attractor-diffsl-llvm/src/lorenz.rs",
]

with open("README.md", "r") as f:
    readme = f.read()

blocks = re.findall(r"```rust\n(.*?)```", readme, re.DOTALL)
if len(blocks) != len(OUTPUTS):
    raise SystemExit(f"expected {len(OUTPUTS)} rust blocks in README.md, found {len(blocks)}")

for code, path in zip(blocks, OUTPUTS):
    with open(path, "w") as f:
        f.write(code)
