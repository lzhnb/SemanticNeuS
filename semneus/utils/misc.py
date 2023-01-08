# Copyright (c) Zhihao Liang. All rights reserved.
import glob
import os
import shutil
import warnings
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def viridis_cmap(gray: np.ndarray) -> np.ndarray:
    """
    Visualize a single-channel image using matplotlib's viridis color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.viridis(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)


def save_img(img: np.ndarray, path: str) -> None:
    """Save an image to disk. Image should have values in [0,1]."""
    img = np.array((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def backup(
    backup_dir: str,
    backup_list: Union[List[str], str],
    contain_suffix: List = ["*.py", "*.h", "*.hpp", "*.cuh", "*.c", "*.cpp", "*.cu"],
    strict: bool = False,
    **kwargs,
) -> None:
    r"""Author: liang.zhihao
    The backup helper function
    Args:
        backup_dir (str): the backup directory
        backup_list (str or List of str): the backup members
        strict (bool, optional): tolerate backup members missing or not.
            Defaults to False.
    """
    # if exist, remove the backup dir to avoid copytree exist error
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    os.makedirs(backup_dir, exist_ok=True)

    print(f"backup files at {backup_dir}")
    if not isinstance(backup_list, list):
        backup_list = [backup_list]
    if not isinstance(contain_suffix, list):
        contain_suffix = [contain_suffix]

    for name in backup_list:
        # deal with missing file or dir
        miss_flag = not os.path.exists(name)
        if miss_flag:
            if strict:
                raise FileExistsError(f"{name} not exist")
            warnings.warn(f"{name} not exist")

        # dangerous dir warning
        if name in ["data", "log"]:
            warnings.warn(f"'{name}' maybe the unsuitable to backup")
        if os.path.isfile(name):
            # just copy the filename
            dst_name = name.split("/")[-1]
            shutil.copy(name, os.path.join(backup_dir, dst_name))
        if os.path.isdir(name):
            # only match '.py' files
            files = glob.iglob(os.path.join(name, "**", "*.*"), recursive=True)
            ignore_suffix = set(map(lambda x: "*." + x.split("/")[-1].split(".")[-1], files))
            for suffix in contain_suffix:
                if suffix in ignore_suffix:
                    ignore_suffix.remove(suffix)
            # copy dir
            shutil.copytree(
                name,
                os.path.join(backup_dir, os.path.basename(name)),
                ignore=shutil.ignore_patterns(*ignore_suffix),
            )
