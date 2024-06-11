import cv2
import numpy as np
import polanalyser as pa


def main():
    # Download data from Hugging Face
    # https://huggingface.co/datasets/jyj7913/spectro-polarimetric

    filename_npy = "spectro-polarimetric/trichromatic/denoised/0449.npy"

    print(f"Load image from {filename_npy}")

    # Load image
    img_rgb_stokes = np.load(filename_npy)  # (H, W, 4, 3)
    img_rgb_stokes = np.moveaxis(img_rgb_stokes, -1, -2)  # (H, W, 3, 4)
    img_bgr_stokes = img_rgb_stokes[..., ::-1, :]

    img_bgr = pa.cvtStokesToIntensity(img_bgr_stokes)

    # Convert stokes parameters
    img_stokes = img_bgr_stokes[..., 1, :]  # Extract (G) green channel
    img_s1 = img_stokes[..., 1]
    img_s2 = img_stokes[..., 2]
    img_s3 = img_stokes[..., 3]
    img_aolp = pa.cvtStokesToAoLP(img_stokes)
    img_dop = pa.cvtStokesToDoP(img_stokes)
    img_dolp = pa.cvtStokesToDoLP(img_stokes)
    img_docp = pa.cvtStokesToDoCP(img_stokes)
    img_ellipticity_angle = pa.cvtStokesToEllipticityAngle(img_stokes)

    # Visualize
    img_bgr_vis = np.clip((img_bgr ** (1 / 2.2)) * 170, 0, 255).astype(np.uint8)
    img_s1_vis = pa.applyColorMap(img_s1, "bwr", -1, 1)
    img_s2_vis = pa.applyColorMap(img_s2, "bwr", -1, 1)
    img_s3_vis = pa.applyColorMap(img_s3, "bwr", -1, 1)
    img_aolp_vis = pa.applyColorToAoLP(img_aolp)
    img_aolp_light_vis = pa.applyColorToAoLP(img_aolp, saturation=img_dolp)
    img_dolp_dark_vis = pa.applyColorToAoLP(img_aolp, value=img_dolp)
    img_top_vis = pa.applyColorToToP(img_ellipticity_angle, img_dop)
    img_cop_vis = pa.applyColorToCoP(img_ellipticity_angle)
    img_dolp_vis = pa.applyColorToDoP(img_dolp)
    img_docp_vis = pa.applyColorToDoP(img_docp)
    img_dop_vis = pa.applyColorToDoP(img_dop)

    cv2.imwrite("color.png", img_bgr_vis)
    cv2.imwrite("s1.png", img_s1_vis)
    cv2.imwrite("s2.png", img_s2_vis)
    cv2.imwrite("s3.png", img_s3_vis)
    cv2.imwrite("aolp.png", img_aolp_vis)
    cv2.imwrite("aolp_light.png", img_aolp_light_vis)
    cv2.imwrite("aolp_dark.png", img_dolp_dark_vis)
    cv2.imwrite("top.png", img_top_vis)
    cv2.imwrite("cop.png", img_cop_vis)
    cv2.imwrite("dolp.png", img_dolp_vis)
    cv2.imwrite("dop.png", img_dop_vis)
    cv2.imwrite("docp.png", img_docp_vis)

    # Colorbar
    w, h = 256, 32

    img_colorbar = (np.linspace(-1, 1, w)[:, None] @ np.ones((1, h))).T
    img_colorbar_bwr = pa.applyColorMap(img_colorbar, "bwr", -1, 1)
    cv2.imwrite("colorbar_bwr.png", img_colorbar_bwr)

    img_colorbar = (np.linspace(0, np.pi, w)[:, None] @ np.ones((1, h))).T
    img_colorbar_aolp = pa.applyColorToAoLP(img_colorbar)
    cv2.imwrite("colorbar_aolp.png", img_colorbar_aolp)

    img_colorbar = (np.linspace(0, 1, w)[:, None] @ np.ones((1, h))).T
    img_colorbar_dolp = pa.applyColorToDoP(img_colorbar)
    cv2.imwrite("colorbar_dolp.png", img_colorbar_dolp)

    img_colorbar = (np.linspace(0, np.pi / 4, w)[:, None] @ np.ones((1, h))).T
    img_colorbar_top = pa.applyColorToToP(img_colorbar)
    cv2.imwrite("colorbar_top.png", img_colorbar_top)

    img_colorbar = (np.linspace(-np.pi / 4, np.pi / 4, w)[:, None] @ np.ones((1, h))).T
    img_colorbar_cop = pa.applyColorToCoP(img_colorbar)
    cv2.imwrite("colorbar_cop.png", img_colorbar_cop)


if __name__ == "__main__":
    main()
