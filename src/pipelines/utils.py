import os
import logging
from typing import List, Tuple
from pathlib import Path

import numpy as np

from PIL import Image

logging.basicConfig(level=logging.DEBUG)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


def label_img(dir_name: str) -> int:
    """Labels images based on their directory name.
    Images in the `hotdog` directory are positive cases,
    while any other dir name is labeled negative.

    Args:
        dir_name (str): Directory name.

    Returns:
        (int): Binary label (1 for hotdog, 0 for not hotdog)
    """
    if dir_name == 'hotdog':
        return 1
    else:
        return 0


def load_data(train: bool,
              configs: dict) -> List:
    """ 
    Returns:
        (list): List with the image array, label array, and filename
    """
    subdir = configs['image_dir_train'] if train else configs['image_dir_test']
    logging.info(f'Loading data from the {subdir} sub-directory.')

    # Resolve relative data directories from the repository root so scripts
    # run correctly from any current working directory.
    image_dir = Path(subdir)
    if not image_dir.is_absolute():
        image_dir = PROJECT_ROOT / image_dir
    image_dir = str(image_dir.resolve())

    data = []
    directories = next(os.walk(image_dir))[1]
    class_to_files = {}

    for dirname in directories:
        class_dir = os.path.join(image_dir, dirname)
        file_names = next(os.walk(class_dir))[2]
        class_to_files[dirname] = [
            image_name for image_name in file_names
            if 'DS_Store' not in image_name and '.csv' not in image_name
        ]

    # Keep training set balanced at 50/50 for hotdog vs not_hotdog.
    if train and 'hotdog' in class_to_files and 'not_hotdog' in class_to_files:
        seed = int(configs.get('random_seed', 23))
        rng = np.random.default_rng(seed)
        min_count = min(len(class_to_files['hotdog']), len(class_to_files['not_hotdog']))
        class_to_files = {
            'hotdog': list(rng.choice(class_to_files['hotdog'], size=min_count, replace=False)),
            'not_hotdog': list(rng.choice(class_to_files['not_hotdog'], size=min_count, replace=False)),
        }
        logging.info('Using balanced training subset: %s hotdog and %s not_hotdog images.', min_count, min_count)

    for dirname, file_names in class_to_files.items():
        logging.info(f'Loading images from the {dirname} directory.')
        for image_name in file_names:
            image_path = os.path.join(image_dir, dirname, image_name)
            label = label_img(dirname)
            img = Image.open(image_path)
            img = img.convert('RGB')
            img = img.resize(
                (configs['image_size'], configs['image_size']),
                RESAMPLE_LANCZOS)
            data.append([np.array(img), label, image_path])
    return data


def format_data_for_model(dat_list: List,
                          configs: dict) -> Tuple:
    """Takes in a list including images as np.ndarrays,
    labels, and image_paths, and reformats them for model
    training/prediction.

    Args:
        dat_list (List[np.ndarray, np.ndarray, np.ndarray])
        configs (dict): Config dictionary

    Returns:
         (np.ndarray, np.ndarray, np.ndarray): formatted data for
         model training or prediction.
    """
    images = np.array(
        [i[0] for i in dat_list]).reshape(
        -1, configs['image_size'], configs['image_size'], 3)
    images = images.astype('float32')
    # The models handle preprocessing, so we don't normalize to 0-1 here

    labels = np.array([i[1] for i in dat_list])

    image_paths = np.array([i[2] for i in dat_list])
    return images, labels, image_paths
