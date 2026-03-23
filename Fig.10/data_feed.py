"""
Dataset construction and loading utilities.

This module generates sample pairs from subfolders under the given root path.
Each sample contains:
    1. input file path list
    2. label file path list

It also provides a Dataset class for loading `.mat` files as tensors.
"""

import os
import random
import torch
import scipy.io as sciio
from natsort import natsorted
from torch.utils.data import Dataset


def build_sample_index(root_dir, use_shuffle, use_natural_sort=False):
    """
    Scan all subfolders under root_dir and build sample path pairs.

    Args:
        root_dir: dataset root directory
        use_shuffle: whether to shuffle the generated sample list
        use_natural_sort: whether to sort subfolder names naturally

    Returns:
        A list of tuples:
            [
                ([input_path1, input_path2, ...], [label_path1, label_path2, ...]),
                ...
            ]
    """
    folder_names = os.listdir(root_dir)

    if use_natural_sort:
        folder_names = natsorted(folder_names)

    indexed_samples = []

    for folder_name in folder_names:
        current_folder = root_dir + '/' + folder_name
        file_names = os.listdir(current_folder)

        if file_names:
            input_file_group = []
            label_file_group = []

            for file_name in file_names:
                name_parts = file_name.split('_')

                if name_parts[0] == 'input':
                    input_file_group.append(current_folder + '/' + str(file_name))
                elif name_parts[0] == 'label':
                    label_file_group.append(current_folder + '/' + str(file_name))

            indexed_samples.append((input_file_group, label_file_group))

    if use_shuffle:
        random.shuffle(indexed_samples)

    return indexed_samples


class MatSequenceDataset(Dataset):
    """
    Dataset for loading paired input/label data from .mat files.
    """

    def __init__(self, data_root, use_natural_sort=False, shuffle_on_init=False):
        self.data_root = data_root
        self.sample_index = build_sample_index(
            self.data_root,
            use_shuffle=shuffle_on_init,
            use_natural_sort=use_natural_sort
        )

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, index):
        input_paths, label_paths = self.sample_index[index]

        for input_path in input_paths:
            input_name = input_path.split('/')
            input_name = input_name[2]
            input_name = input_name.split('.')
            input_name = input_name[0]

            input_tensor = sciio.loadmat(input_path)[input_name]
            input_tensor = torch.as_tensor(input_tensor)

        for label_path in label_paths:
            label_name = label_path.split('/')
            label_name = label_name[2]
            label_name = label_name.split('.')
            label_name = label_name[0]

            label_tensor = sciio.loadmat(label_path)[label_name]
            label_tensor = torch.as_tensor(label_tensor)

        return input_tensor, label_tensor
