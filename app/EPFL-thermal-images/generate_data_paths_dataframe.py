from os.path import join, relpath, dirname, abspath, basename
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

from glob import glob

DEFAULT_DIR = abspath(
    join(
        dirname(relpath(__file__)),
        "..",
        "..",
        "data",
        "EPFL-thermal-images",
        "nirscene1",
    )
)


def generate_train_test_dataframes_for_epfl_dataset(root_dir):
    folder_paths = glob(f"{root_dir}/*")

    print(f"Folders: {folder_paths}")

    nir_img_paths = []

    for path in folder_paths:
        nir_img_paths += sorted(glob(f"{path}/*nir.tiff"))
        nir_img_paths = [abspath(path) for path in nir_img_paths]

    rgb_img_paths = [
        nir_path.replace("nir.tiff", "rgb.tiff") for nir_path in nir_img_paths
    ]

    labels = [basename(dirname(nir_path)) for nir_path in nir_img_paths]

    data = {"rgb": rgb_img_paths, "infrared": nir_img_paths, "label": labels}
    df = pd.DataFrame(data=data)

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df.label
    )

    save_path_train = abspath(join(dirname(relpath(__file__)), f"train.json"))

    save_path_val = abspath(join(dirname(relpath(__file__)), f"val.json"))

    train_df.to_json(save_path_train)
    val_df.to_json(save_path_val)

    print(
        f"Saved train dataset in {save_path_train} with {len(train_df)} annotations")
    print(
        f"Saved val dataset in {save_path_val} with {len(val_df)} annotations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="parser_format_data_generator.py",
        description=(
            "Generate dataframe with paths to images and masks \n"
            + " " * 4
            + "python generate_data_paths_dataframe.py path/to/EPFL/dir"
        ),
    )

    parser.add_argument(
        "--root_dir", default=DEFAULT_DIR, help="Path to the PST900 root dir"
    )

    args = parser.parse_args()

    generate_train_test_dataframes_for_epfl_dataset(args.root_dir)
