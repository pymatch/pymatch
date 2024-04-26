#!/usr/bin/env bash

# With and without fastmath and parallel on fastest version?

python harness.py numpy "" --mnist-size=10 --print-header
python harness.py numpy "" --mnist-size=100
python harness.py numpy "" --mnist-size=1000
python harness.py numpy "" --mnist-size=60000

python harness.py torch "" --mnist-size=10
python harness.py torch "" --mnist-size=100
python harness.py torch "" --mnist-size=1000
python harness.py torch "" --mnist-size=60000

MATRIX_BACKEND=loops NUMBA_DISABLE_JIT=1 python harness.py matrix "loops.nojit" --mnist-size=10
MATRIX_BACKEND=loops NUMBA_DISABLE_JIT=1 python harness.py matrix "loops.nojit" --mnist-size=100
MATRIX_BACKEND=loops NUMBA_DISABLE_JIT=1 python harness.py matrix "loops.nojit" --mnist-size=1000
MATRIX_BACKEND=loops NUMBA_DISABLE_JIT=1 python harness.py matrix "loops.nojit" --mnist-size=60000

MATRIX_BACKEND=loops NUMBA_DISABLE_JIT=0 python harness.py matrix "loops.jit" --mnist-size=10
MATRIX_BACKEND=loops NUMBA_DISABLE_JIT=0 python harness.py matrix "loops.jit" --mnist-size=100
MATRIX_BACKEND=loops NUMBA_DISABLE_JIT=0 python harness.py matrix "loops.jit" --mnist-size=1000
MATRIX_BACKEND=loops NUMBA_DISABLE_JIT=0 python harness.py matrix "loops.jit" --mnist-size=60000

MATRIX_BACKEND=listcomp NUMBA_DISABLE_JIT=1 python harness.py matrix "listcomp-nojit" --mnist-size=10
MATRIX_BACKEND=listcomp NUMBA_DISABLE_JIT=1 python harness.py matrix "listcomp-nojit" --mnist-size=100
MATRIX_BACKEND=listcomp NUMBA_DISABLE_JIT=1 python harness.py matrix "listcomp-nojit" --mnist-size=1000
MATRIX_BACKEND=listcomp NUMBA_DISABLE_JIT=1 python harness.py matrix "listcomp-nojit" --mnist-size=60000

MATRIX_BACKEND=listcomp NUMBA_DISABLE_JIT=0 python harness.py matrix "listcomp-jit" --mnist-size=10
MATRIX_BACKEND=listcomp NUMBA_DISABLE_JIT=0 python harness.py matrix "listcomp-jit" --mnist-size=100
MATRIX_BACKEND=listcomp NUMBA_DISABLE_JIT=0 python harness.py matrix "listcomp-jit" --mnist-size=1000
MATRIX_BACKEND=listcomp NUMBA_DISABLE_JIT=0 python harness.py matrix "listcomp-jit" --mnist-size=60000

python harness.py extension "" --mnist-size=10
python harness.py extension "" --mnist-size=100
python harness.py extension "" --mnist-size=1000
python harness.py extension "" --mnist-size=60000
