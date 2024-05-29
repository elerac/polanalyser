from sympy import symbols
from sympy import sin, cos
from sympy import Matrix
from sympy import pi


def mueller(symbol="m"):
    m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33 = symbols(f"{symbol}:4:4", real=True)
    return Matrix([[m00, m01, m02, m03], [m10, m11, m12, m13], [m20, m21, m22, m23], [m30, m31, m32, m33]])


def polarizer(theta):
    s = sin(2 * theta)
    c = cos(2 * theta)
    return 0.5 * Matrix(
        [
            [1, c, s, 0],
            [c, c**2, c * s, 0],
            [s, c * s, s**2, 0],
            [0, 0, 0, 0],
        ]
    )


def rotator(theta):
    s = sin(2 * theta)
    c = cos(2 * theta)
    return Matrix(
        [
            [1, 0, 0, 0],
            [0, c, s, 0],
            [0, -s, c, 0],
            [0, 0, 0, 1],
        ]
    )


def retarder(delta, theta):
    s = sin(delta)
    c = cos(delta)
    m = Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, s],
            [0, 0, -s, c],
        ]
    )
    return rotator(-theta) @ m @ rotator(theta)


def qwp(theta):
    return retarder(pi / 2, theta)


def hwp(theta):
    return retarder(pi, theta)
