from sympy import symbols
from sympy import atan, sqrt, Abs
from sympy import Matrix


def stokes(symbol="s"):
    s0, s1, s2, s3 = symbols(f"{symbol}:4", real=True)
    return Matrix([s0, s1, s2, s3])


def cvtStokesToAoLP(stokes):
    s0, s1, s2, s3 = stokes
    return 0.5 * atan(s2 / s1)


def cvtStokesToDoLP(stokes):
    s0, s1, s2, s3 = stokes
    return sqrt(s1**2 + s2**2) / s0


def cvtStokesToDoCP(stokes):
    s0, s1, s2, s3 = stokes
    return Abs(s3) / s0


def cvtStokesToDoP(stokes):
    s0, s1, s2, s3 = stokes
    return sqrt(s1**2 + s2**2 + s3**2) / s0
