"""Correctness and performance harness for PyFR's batched LU inverse.

Runs candidate implementations of `LUInverse` through PyFR's real backend,
exactly as the block-Jacobi preconditioner calls them.  See README.md for
usage.
"""
