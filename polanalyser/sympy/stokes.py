from sympy import Abs, Eq, Matrix, Mod, Piecewise, atan2, pi, sqrt, symbols


def stokes(symbol="s"):
    s0, s1, s2, s3 = symbols(f"{symbol}:4", real=True)
    return Matrix([s0, s1, s2, s3])


def stokes_to_aolp(stokes):
    s0, s1, s2, s3 = stokes
    aolp = Mod(0.5 * atan2(s2, s1), pi)
    return Piecewise((0, Eq(s1, 0) & Eq(s2, 0)), (aolp, True))


def stokes_to_dolp(stokes):
    s0, s1, s2, s3 = stokes
    return sqrt(s1**2 + s2**2) / s0


def stokes_to_docp(stokes):
    s0, s1, s2, s3 = stokes
    return Abs(s3) / s0


def stokes_to_dop(stokes):
    s0, s1, s2, s3 = stokes
    return sqrt(s1**2 + s2**2 + s3**2) / s0


def stokes_to_eang(stokes):
    s0, s1, s2, s3 = stokes
    eang = 0.5 * atan2(s3, sqrt(s1**2 + s2**2))
    return Piecewise((0, Eq(s1, 0) & Eq(s2, 0) & Eq(s3, 0)), (eang, True))
