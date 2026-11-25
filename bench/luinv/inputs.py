import numpy as np


def _jacobian(rng, k, n, rho=0.9):
    # Shifted Gaussian, like I - gamma*dt*J with a moderate spectral radius
    return np.eye(n) - rho/np.sqrt(n)*rng.standard_normal((k, n, n))


def _nopivot(rng, k, n):
    # Strictly column diagonally dominant so partial pivoting never swaps
    a = rng.standard_normal((k, n, n))
    i = np.arange(n)
    a[:, i, i] = 0
    a[:, i, i] = 1.1*np.abs(a).sum(axis=1)*rng.choice([-1, 1], (k, n))

    return a


def _pivot(rng, k, n):
    # Rows of a pivot free matrix shuffled so nearly every column pivots
    a = _nopivot(rng, k, n)
    for m in a:
        m[:] = m[rng.permutation(n)]

    return a


def _dense(rng, k, n):
    # Unstructured Gaussian matrices; condition numbers of O(n) and above
    return rng.standard_normal((k, n, n)) / np.sqrt(n)


def _illcond(rng, k, n, cond=1e6, npool=4):
    # Few orthogonally similar matrices of the given condition number, made
    # distinct per element through a scale factor; the singular values are
    # centred on one so the inverses stay in range of half precision
    s = np.logspace(np.log10(cond) / 2, -np.log10(cond) / 2, n)
    pool = []
    for _ in range(min(k, npool)):
        u, _ = np.linalg.qr(rng.standard_normal((n, n)))
        v, _ = np.linalg.qr(rng.standard_normal((n, n)))
        pool.append((u*s) @ v.T)

    f = rng.uniform(0.5, 2, k)
    return np.array([fi*pool[i % len(pool)] for i, fi in enumerate(f)])


KINDS = {
    'jacobian': _jacobian,
    'nopivot': _nopivot,
    'pivot': _pivot,
    'dense': _dense,
    'illcond': _illcond,
}


def make_jacobians(kind, k, n, scale, seed=0):
    """Jacobians J for which I - scale*J is a matrix of the given kind."""
    try:
        gen = KINDS[kind]
    except KeyError:
        raise ValueError(f'Unknown input kind {kind!r}') from None

    a = gen(np.random.default_rng(seed), k, n)
    return (np.eye(n) - a) / scale


def load_jacobians(path, n):
    # Jacobians dumped from a PyFR run as an (n, n, k) staging array
    jac = np.load(path)
    if jac.shape[:2] != (n, n):
        raise ValueError(f'Expected {n}x{n} Jacobians in {path}')

    return np.ascontiguousarray(jac.transpose(2, 0, 1), dtype=np.float64)
