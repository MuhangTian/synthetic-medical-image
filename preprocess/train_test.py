import os

import numpy as np
from tqdm import tqdm


def train_test_split(image_path, test_size=0.2, seed=2023):
    """make train and test folders and move images to them"""
    list_dir = os.listdir(image_path)
    indices = np.arange(1, len(list_dir)+1)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    test_indices = indices[:int(test_size * len(list_dir))]
    train_indices = indices[int(test_size * len(list_dir)):]
    
    for test_idx in tqdm(test_indices, desc="Moving test images..."):
        if not os.path.exists(os.path.join(image_path, "test")):
            os.makedirs(os.path.join(image_path, "test"))
        os.rename(os.path.join(image_path, f"{test_idx}.png"), os.path.join(image_path, "test", f"{test_idx}.png"))
    
    for train_idx in tqdm(train_indices, desc="Moving train images..."):
        if not os.path.exists(os.path.join(image_path, "train")):
            os.makedirs(os.path.join(image_path, "train"))
        os.rename(os.path.join(image_path, f"{train_idx}.png"), os.path.join(image_path, "train", f"{train_idx}.png"))
    