"""Reference points for batched inversion on HIP: rocSOLVER's batched LU
inverse, a batched GEMM of the same nominal FLOPs, and a device copy of the
same compulsory bytes."""

from ctypes import POINTER, c_double, c_float, c_int, c_int64, c_void_p

import numpy as np

from pyfr.ctypesutil import LibWrapper

from bench.luinv.harness import nominal_flops
from bench.luinv.timing import stats


class RocError(Exception): pass


_rocsolver_fns = []
for t in 'ds':
    _rocsolver_fns += [
        (c_int, f'rocsolver_{t}getrf_strided_batched', c_void_p, c_int, c_int,
         c_void_p, c_int, c_int64, c_void_p, c_int64, c_void_p, c_int),
        (c_int, f'rocsolver_{t}getri_outofplace_strided_batched', c_void_p,
         c_int, c_void_p, c_int, c_int64, c_void_p, c_int64, c_void_p, c_int,
         c_int64, c_void_p, c_int)
    ]


class RocSOLVERWrappers(LibWrapper):
    _libname = 'rocsolver'
    _statuses = {'*': RocError}
    _functions = _rocsolver_fns


class RocBLASWrappers(LibWrapper):
    _libname = 'rocblas'
    _statuses = {'*': RocError}

    OPERATION_NONE = 111

    _functions = [
        (c_int, 'rocblas_create_handle', POINTER(c_void_p)),
        (c_int, 'rocblas_set_stream', c_void_p, c_void_p),
        *[(c_int, f'rocblas_{t}gemm_strided_batched', c_void_p, c_int, c_int,
           c_int, c_int, c_int, POINTER(ct), c_void_p, c_int, c_int64,
           c_void_p, c_int, c_int64, POINTER(ct), c_void_p, c_int, c_int64,
           c_int) for t, ct in [('d', c_double), ('s', c_float)]]
    ]


class References:
    def __init__(self, backend):
        if backend.name != 'hip':
            raise RuntimeError('References need the HIP backend')

        self.backend = backend
        self.hip = hip = backend.hip
        self.stream = backend._stream

        self.blas = RocBLASWrappers()
        self.solver = RocSOLVERWrappers()

        # rocSOLVER takes a rocBLAS handle bound to our stream
        self.handle = c_void_p()
        self.blas.rocblas_create_handle(self.handle)
        self.blas.rocblas_set_stream(self.handle, self.stream)

        self.evts = hip.create_event(), hip.create_event()

    def _alloc(self, ary):
        buf = self.hip.mem_alloc(ary.nbytes)
        self.hip.memcpy(buf, ary, ary.nbytes)
        return buf

    def _time(self, fn, pre=None, reps=20, warmup=2):
        start, stop = self.evts
        times = []
        for i in range(warmup + reps):
            if pre:
                pre()

            start.record(self.stream)
            fn()
            stop.record(self.stream)
            stop.synchronize()

            if i >= warmup:
                times.append(stop.elapsed_time(start))

        return stats(np.array(times))

    def _matrices(self, n, neles, dtype, seed=0):
        # Well conditioned matrices, column-major and densely packed
        rng = np.random.default_rng(seed)
        a = np.eye(n) - 0.9/np.sqrt(n)*rng.standard_normal((neles, n, n))
        return a.astype(dtype)

    def rocsolver(self, n, neles, dtype, reps=20, check=True):
        dtype = np.dtype(dtype)
        t = 'd' if dtype == np.float64 else 's'
        getrf = getattr(self.solver, f'rocsolver_{t}getrf_strided_batched')
        getri = getattr(self.solver,
                        f'rocsolver_{t}getri_outofplace_strided_batched')

        a0 = self._matrices(n, neles, dtype)
        da0, da = self._alloc(a0), self._alloc(a0)
        dc = self.hip.mem_alloc(a0.nbytes)
        ipiv = self.hip.mem_alloc(4*n*neles)
        info = self.hip.mem_alloc(4*neles)

        # Factorise a fresh copy each time and invert out of place
        def pre():
            self.hip.memcpy(da, da0, a0.nbytes, self.stream)

        def run():
            getrf(self.handle, n, n, da, n, n*n, ipiv, n, info, neles)
            getri(self.handle, n, da, n, n*n, ipiv, n, dc, n, n*n, info,
                  neles)

        res = self._time(run, pre, reps)

        if check:
            # The transposed matrices invert to the transposed inverses so
            # the row-major reading of the column-major data is consistent
            c = np.empty_like(a0)
            self.hip.memcpy(c, dc, c.nbytes)
            err = np.abs(np.linalg.inv(a0[:4].astype(np.float64)) - c[:4])
            res['max_abs_err'] = float(err.max())

        return self._summarise('rocsolver_getrf_getri', n, neles, dtype, res,
                               flops=nominal_flops(n)*neles)

    def gemm(self, n, neles, dtype, reps=20):
        dtype = np.dtype(dtype)
        t, ct = ('d', c_double) if dtype == np.float64 else ('s', c_float)
        fn = getattr(self.blas, f'rocblas_{t}gemm_strided_batched')

        a = self._matrices(n, neles, dtype)
        da, db = self._alloc(a), self._alloc(a)
        dc = self.hip.mem_alloc(a.nbytes)
        one, zero, op = ct(1), ct(0), self.blas.OPERATION_NONE

        # One n^3 GEMM per element has the nominal FLOPs of an inverse
        def run():
            fn(self.handle, op, op, n, n, n, one, da, n, n*n, db, n, n*n,
               zero, dc, n, n*n, neles)

        res = self._time(run, reps=reps)
        return self._summarise('rocblas_gemm', n, neles, dtype, res,
                               flops=nominal_flops(n)*neles)

    def copy(self, nbytes, reps=20):
        src = self.hip.mem_alloc(nbytes)
        dst = self.hip.mem_alloc(nbytes)

        res = self._time(lambda: self.hip.memcpy(dst, src, nbytes,
                                                 self.stream), reps=reps)
        res['gbs'] = 2*nbytes / res['median_s'] / 1e9
        return {'mode': 'ref', 'ref': 'copy', 'nbytes': nbytes, **res}

    def _summarise(self, name, n, neles, dtype, res, flops):
        res['tflops'] = flops / res['median_s'] / 1e12
        res['matrices_per_s'] = neles / res['median_s']

        return {'mode': 'ref', 'ref': name, 'n': n, 'neles': neles,
                'dtype': dtype.name, **res}
