import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import polanalyser as pa


def save_wheel_reference_color(filename: str, is_saturation: bool = False, is_value: bool = False) -> None:
    # Create a background image
    res = 1024
    radius = 0.5 * res
    img_r = np.empty((res, res), dtype=np.float64)
    img_theta = np.empty((res, res), dtype=np.float64)
    img_mask = np.empty((res, res), dtype=np.uint8)
    for j in range(res):
        for i in range(res):
            x = i - 0.5 * res
            y = -(j - 0.5 * res)
            r = np.sqrt(x**2 + y**2) / radius
            theta = np.arctan2(y, x)

            img_r[j, i] = r
            img_theta[j, i] = np.mod(theta, np.pi)

    # Mask of in/out wheel
    img_inside_wheel = img_r < 1
    img_outside_wheel = np.logical_not(img_inside_wheel)

    # Convert theta and r to colored image
    if is_saturation == False and is_value == False:
        saturation = 1.0
        value = 1.0
    elif is_saturation == True and is_value == False:
        saturation = img_r
        value = 1.0
    elif is_saturation == False and is_value == True:
        saturation = 1.0
        value = img_r
    else:
        raise ValueError

    img_wheel_rgb = pa.applyColorToAoLP(img_theta, saturation, value)[:, :, ::-1]

    # Alpha
    img_wheel_alpha = np.empty((res, res), dtype=np.uint8)
    img_wheel_alpha[img_inside_wheel] = 255
    img_wheel_alpha[img_outside_wheel] = 0  # Transparent background

    # Merge RGB and Alpha
    img_wheel_rgba = np.empty((res, res, 4), dtype=np.uint8)
    img_wheel_rgba[:, :, :3] = img_wheel_rgb
    img_wheel_rgba[:, :, 3] = img_wheel_alpha

    # Plot
    fig = plt.figure()

    # Plot background image
    ax0 = fig.add_subplot(111)
    ax0.imshow(img_wheel_rgba)
    ax0.axis("off")

    # Plor polar axis
    ax1 = fig.add_subplot(111, polar=True)
    ax1.grid(False, axis="both")  # Delete grid
    ax1.set_yticks([])
    # ax1.set_yticks([0, 0.5, 1.0])

    # Set stroke
    text_color = matplotlib.colors.hex2color(plt.rcParams["text.color"])
    foreground_color = tuple(1 - np.array(text_color))
    path_effects = [withStroke(linewidth=1, foreground=foreground_color)]

    for label in ax1.get_xticklabels():
        label.set_path_effects(path_effects)

    for label in ax1.get_yticklabels():
        label.set_path_effects(path_effects)

    # Save
    fig.savefig(filename, bbox_inches="tight", transparent=True)
    plt.close("all")


def main():
    # Output filename settings
    name = "AoLP_wheel"
    ext = ".svg"

    style_list = ["default", "dark_background"]
    for style in style_list:
        for is_saturation in [False, True]:
            for is_value in [False, True]:
                if is_saturation and is_value:
                    continue

                # Matplotlib settings
                plt.style.use(style)
                plt.rcParams["font.size"] = 14
                plt.rcParams["font.weight"] = "normal"

                # Filename
                if is_saturation == False and is_value == False:
                    modulation_type = ""
                elif is_saturation == True and is_value == False:
                    modulation_type = "_saturation"
                elif is_saturation == False and is_value == True:
                    modulation_type = "_value"

                if style == "default":
                    style_type = "_light"
                elif style == "dark_background":
                    style_type = "_dark"

                filename = f"{name}{modulation_type}{style_type}{ext}"

                # Generate wheel and save
                print(f"Export '{filename}'")
                save_wheel_reference_color(filename, is_saturation, is_value)

    print("Done!")


if __name__ == "__main__":
    main()
