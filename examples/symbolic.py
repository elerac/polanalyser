from sympy import symbols, simplify
import polanalyser.sympy as pas


def main():
    theta = symbols("theta", real=True)

    # Example 1: Malus's law
    M_L1 = pas.polarizer(0)
    M_L2 = pas.polarizer(theta)
    f = (M_L2 @ M_L1)[0, 0]
    print(simplify(f))
    # -> 0.5*cos(theta)**2

    # Example 2: Ellipsometer [Azzam+, 1978][Baek+, 2020]
    M = pas.mueller()  # Symbolic Mueller matrix
    I = 1.0
    M_PSG = pas.qwp(theta) @ pas.polarizer(0)
    M_PSA = pas.polarizer(0) @ pas.qwp(5 * theta)
    f = (M_PSA @ M @ M_PSG * I)[0, 0]
    print(simplify(f))
    # -> 0.25*m00 + 0.25*m10*cos(10*theta)**2 + 0.125*m20*sin(20*theta) - 0.25*m30*sin(10*theta) + 0.25*(m01 + m11*cos(10*theta)**2 + m21*sin(20*theta)/2 - m31*sin(10*theta))*cos(2*theta)**2 + 0.25*(m02 + m12*cos(10*theta)**2 + m22*sin(20*theta)/2 - m32*sin(10*theta))*sin(2*theta)*cos(2*theta) + 0.25*(m03 + m13*cos(10*theta)**2 + m23*sin(20*theta)/2 - m33*sin(10*theta))*sin(2*theta)


if __name__ == "__main__":
    main()
