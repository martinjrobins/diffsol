# a python script that extracts the single code block from README.md
# and writes it to examples/lorenz-attractor-diffsl-llvm/src/lorenz.rs.
# This is used in a github workflow to keep the example code in sync
import re

with open("README.md", "r") as f:
    readme = f.read()

match = re.search(r"```rust\n(.*?)```", readme, re.DOTALL)
if match:
    code = match.group(1)
    with open("examples/lorenz-attractor-diffsl-llvm/src/lorenz.rs", "w") as f:
        f.write(code)