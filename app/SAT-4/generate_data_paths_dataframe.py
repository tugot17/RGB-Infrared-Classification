from os.path import join, relpath, dirname, abspath, basename
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import scipy.io

from glob import glob

DEFAULT_SAT_4_PATH = abspath(
    join(
        dirname(relpath(__file__)),
        "..",
        "..",
        "data",
        "SAT-4_and_SAT-6_datasets",
        "sat-4-full.mat",
    )
)

DEFAULT_IMAGE_SAVE_DIR_PATH = abspath(
    join(
        dirname(relpath(__file__)),
        "..",
        "..",
        "data",
        "SAT_4",
    )
)


def main(sat_4_full_mat_path, images_save_dir_path):
    mat = scipy.io.loadmat(sat_4_full_mat_path)

    class_to_label_map = {0: "barren land", 1: "trees", 2: "grassland", 3: "none"}

    from tqdm.auto import tqdm
    from PIL import Image
    from os.path import join

    stages = ["train", "test"]

    for stage in stages:
        rgb_img_paths = []
        nir_img_paths = []
        labels = []

        for idx in tqdm(range(mat[f"{stage}_x"].shape[-1])):
            rgb_save_path = abspath(relpath(join(images_save_dir_path, stage, "rgb", f"{idx}.png")))
            nir_path_rgb = abspath(relpath(join(images_save_dir_path, stage, "infrared", f"{idx}.png")))

            rgb = mat[f"{stage}_x"][:, :, :3, idx]
            rgb = Image.fromarray(rgb)
            rgb.save(rgb_save_path, "PNG")

            nir = mat[f"{stage}_x"][:, :, 3, idx]
            nir = Image.fromarray(nir)
            nir.save(nir_path_rgb, "PNG")

            label = class_to_label_map[mat[f"{stage}_y"][:, idx].argmax()]

            rgb_img_paths.append(rgb_save_path)
            nir_img_paths.append(nir_path_rgb)
            labels.append(label)

        data = {"rgb": rgb_img_paths, "infrared": nir_img_paths, "label": labels}
        df = pd.DataFrame(data=data)

        json_dataframe_path = abspath(join(dirname(relpath(__file__)), f"{stage}.json"))
        df.to_json(json_dataframe_path)

        print(
            f"Saved {stage} dataset in {json_dataframe_path} with {len(df)} annotations"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="parser_format_data_generator.py",
        description=(
            "Generate dataframe with paths to images and masks \n"
            + " " * 4
            + "python generate_data_paths_dataframe.py ../..data/SAT-4_and_SAT-6_datasets/sat-4-full.mat"
        ),
    )

    parser.add_argument(
        "--sat_4_full_mat_path",
        default=DEFAULT_SAT_4_PATH,
        help="Path to the sat-4-full.mat file",
    )

    parser.add_argument(
        "--images_save_dir_path",
        default=DEFAULT_IMAGE_SAVE_DIR_PATH,
        help="Dir in which you want to save generated images",
    )

    args = parser.parse_args()

    main(args.sat_4_full_mat_path, args.images_save_dir_path)
