import argparse
import numpy as np
import cv2
import polanalyser as pa


def mueller_image(M: np.ndarray, colormap: str = "RdBu", size: int = 64, border: int = 2) -> np.ndarray:
    if M.shape not in [(3, 3), (4, 4)]:
        raise ValueError("Mueller matrix must be 3x3 or 4x4.")

    # Apply color map
    M_vis = pa.applyColorMap(M, colormap, -M[0, 0], M[0, 0])

    # Repeat for 64x64 blocks
    img_M_vis = M_vis[None, None, ...]  # (1, 1, 4, 4, 3)
    img_M_vis = np.repeat(img_M_vis, size, axis=0)
    img_M_vis = np.repeat(img_M_vis, size, axis=1)

    # Make grid
    img_M_vis_grid = pa.makeGridMueller(img_M_vis, border=border)

    return img_M_vis_grid


def main():
    np.set_printoptions(precision=2, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mueller", type=str, default="pa.polarizer(3.14 / 2)", help="Mueller matrix as evaluable string. e.g. 'pa.polarizer(3.14 / 2)', 'np.eye(4)', '[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]'")
    parser.add_argument("--colormap", type=str, default="RdBu", help="Colormap name.")
    parser.add_argument("--size", type=int, default=64, help="Size of each Mueller matrix block.")
    parser.add_argument("--border", type=int, default=2, help="Border size between each Mueller matrix block.")
    parser.add_argument("-o", "--output", type=str, default="mueller_vis.png", help="Output file path.")
    args = parser.parse_args()

    print("Arguments:")
    print(args)

    # Mueller matrix (3x3 or 4x4)
    M = np.array(eval(args.mueller))

    print(f"Mueller matrix {M.shape}:")
    print(M)

    img_M_vis_grid = mueller_image(M, args.colormap, args.size, args.border)

    print(f"Exporting to {args.output} {img_M_vis_grid.shape}")
    cv2.imwrite(args.output, img_M_vis_grid)


if __name__ == "__main__":
    main()
