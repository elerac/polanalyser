import polanalyser as pa

import numpy as np

np.set_printoptions(precision=3, suppress=True)


def main():
    # Polarizer
    M_LP = pa.polarizer(0)
    print(M_LP)
    # [[0.5 0.5 0.  0. ]
    #  [0.5 0.5 0.  0. ]
    #  [0.  0.  0.  0. ]
    #  [0.  0.  0.  0. ]]

    # Quarter-wave plate
    M_QWP = pa.qwp(0)
    print(M_QWP)
    # [[ 1.  0.  0.  0.]
    #  [ 0.  1.  0.  0.]
    #  [ 0.  0.  0.  1.]
    #  [ 0.  0. -1.  0.]]


if __name__ == "__main__":
    main()
