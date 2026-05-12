## Test entry point used by `Pkg.test` and CI.
##
## All test logic lives in `unit_tests.jl` — keep this file as a thin includer so
## adding/removing testsets stays a one-file change. The legacy script-style
## tests that previously lived here have been retired (they relied on benchmark
## CFRadial files and pre-1.2.0 keyword names like `NCP_THRESHOLD`); their
## coverage of feature math, weighted reductions, and write_field is preserved
## in unit_tests.jl.

using Test
using Ronin
include("unit_tests.jl")
