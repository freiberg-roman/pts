import os
import shutil
from os import listdir
from os.path import isfile, join
from pathlib import Path


def merge_data(base_path, val_percentage=0.03, test_percentage=0.02):
    subfolders = [
        f.path
        for f in os.scandir(base_path)
        if f.is_dir()
        and f.path.split("/")[-1] != "train"
        and f.path.split("/")[-1] != "val"
        and f.path.split("/")[-1] != "test"
    ]
    for type in ["depth", "seg", "rgb"]:
        Path(base_path + "train/" + type).mkdir(parents=True, exist_ok=True)
        Path(base_path + "val/" + type).mkdir(parents=True, exist_ok=True)
        Path(base_path + "test/" + type).mkdir(parents=True, exist_ok=True)

    counter = {
        "train": len(
            [
                f
                for f in listdir(base_path + "train/depth/")
                if isfile(join(base_path + "train/depth/", f))
            ]
        ),
        "val": len(
            [
                f
                for f in listdir(base_path + "val/depth/")
                if isfile(join(base_path + "val/depth/", f))
            ]
        ),
        "test": len(
            [
                f
                for f in listdir(base_path + "test/depth/")
                if isfile(join(base_path + "test/depth/", f))
            ]
        ),
    }
    global_counter = counter["train"] + counter["val"] + counter["test"]
    for subfolder in subfolders:
        depth_files = [
            f
            for f in listdir(subfolder + "/depth/")
            if isfile(join(subfolder + "/depth/", f))
        ]
        seg_files = [
            f
            for f in listdir(subfolder + "/seg/")
            if isfile(join(subfolder + "/seg/", f))
        ]
        rgb_files = [
            f
            for f in listdir(subfolder + "/rgb/")
            if isfile(join(subfolder + "/rgb/", f))
        ]
        depth_files.sort()
        seg_files.sort()
        rgb_files.sort()

        for depth, seg, rgb in zip(depth_files, seg_files, rgb_files):
            if counter["train"] == 0:
                division = "train"
            elif counter["val"] / global_counter < val_percentage:
                division = "val"
            elif counter["test"] / global_counter < test_percentage:
                division = "test"
            else:
                division = "train"

            os.rename(
                subfolder + "/depth/" + depth,
                base_path + division + "/depth/" + str(counter[division]) + ".npy",
            )
            os.rename(
                subfolder + "/seg/" + seg,
                base_path + division + "/seg/" + str(counter[division]) + ".npy",
            )
            os.rename(
                subfolder + "/rgb/" + rgb,
                base_path + division + "/rgb/" + str(counter[division]) + ".png",
            )

            counter[division] += 1
            global_counter += 1

        shutil.rmtree(subfolder)
