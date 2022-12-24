"""Calculate Mueller matrix from images captured under a variety of polarimetric conditions.

Download dataset from here.
- https://drive.google.com/drive/folders/1W66tMue6xi0F1QSG9_sDEgeukV-8QkZs?usp=sharing
"""
import argparse
import polanalyser as pa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="dataset/toy_example_3x3_pc")
    args = parser.parse_args()

    path = args.input
    print(f"Load images from '{path}'")
    pcontainer = pa.PolarizationContainer(path)

    print("Calculate Mueller matrix")
    image_list = pcontainer.get_list("image")
    mueller_psg_list = pcontainer.get_list("mueller_psg")
    mueller_psa_list = pcontainer.get_list("mueller_psa")
    img_mueller = pa.calcMueller(image_list, mueller_psg_list, mueller_psa_list)

    print(img_mueller.shape, img_mueller.dtype)

    # Normalized by m00 for visualization
    img_mueller_normalized = img_mueller.copy()
    n1, n2 = img_mueller_normalized.shape[-2:]
    img_mueller_normalized_m00 = img_mueller_normalized[..., 0, 0]
    for i in range(n2):
        for j in range(n1):
            if i == 0 and j == 0:
                continue
            img_mueller_normalized[..., j, i] /= img_mueller_normalized_m00

    filename_visalize = "plot_mueller.png"
    filename_visalize_normalized = "plot_mueller_normalized.png"
    print(f"Visualize and export to '{filename_visalize}' and '{filename_visalize_normalized}'")
    pa.plotMueller(filename_visalize, img_mueller, vabsmax=2.0)
    pa.plotMueller(filename_visalize_normalized, img_mueller_normalized, vabsmax=1.0)


if __name__ == "__main__":
    main()
