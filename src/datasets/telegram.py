# The code in this file is taken from PyTorch Geometric Signed Directed (https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/)

from typing import Optional, Callable

import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data, InMemoryDataset, download_url

from typing import Optional, Union, List

import torch
import numpy as np
from torch_geometric.data import Data


def node_class_split(
    data: Data,
    train_size: Union[int, float] = None,
    val_size: Union[int, float] = None,
    test_size: Union[int, float] = None,
    seed_size: Union[int, float] = None,
    train_size_per_class: Union[int, float] = None,
    val_size_per_class: Union[int, float] = None,
    test_size_per_class: Union[int, float] = None,
    seed_size_per_class: Union[int, float] = None,
    seed: List[int] = [],
    data_split: int = 10,
) -> Data:
    r"""Train/Val/Test/Seed split for node classification tasks.
    The size parameters can either be int or float.
    If a size parameter is int, then this means the actual number, if it is float, then this means a ratio.
    ``train_size`` or ``train_size_per_class`` is mandatory, with the former regardless of class labels.
    Validation and seed masks are optional. Seed masks here masks nodes within the training set, e.g., in a semi-supervised setting as described in the
    `SSSNET: Semi-Supervised Signed Network Clustering <https://arxiv.org/pdf/2110.06623.pdf>`_ paper.
    If test_size and test_size_per_class are both None, all the remaining nodes after selecting training (and validation) nodes will be included.

    Arg types:
        * **data** (Data or DirectedData, required) - The data object for data split.
        * **train_size** (int or float, optional) - The size of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **val_size** (int or float, optional) - The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size** (int or float, optional) - The size of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size** (int or float, optional) - The size of random splits for the seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **train_size_per_class** (int or float, optional) - The size per class of random splits for the training dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **val_size_per_class** (int or float, optional) - The size per class of random splits for the validation dataset. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **test_size_per_class** (int or float, optional) - The size per class of random splits for the testing dataset. If the input is a float number, the ratio of nodes in each class will be sampled. (Default: None. All nodes not selected for training/validation are used for testing)
        * **seed_size_per_class** (int or float, optional) - The size per class of random splits for seed nodes within the training set. If the input is a float number, the ratio of nodes in each class will be sampled.
        * **seed** (An empty list or a list with the length of data_split, optional) - The random seed list for each data split.
        * **data_split** (int, optional) - number of splits (Default : 10)

    Return types:
        * **data** (Data or DirectedData) - The data object includes train_mask, val_mask and test_mask.
    """
    if train_size is None and train_size_per_class is None:
        raise ValueError("Please input the values of train_size or train_size_per_class!")

    if seed_size is not None and seed_size_per_class is not None:
        raise Warning(
            "The seed_size_per_class will be considered if both seed_size and seed_size_per_class are given!"
        )
    if test_size is not None and test_size_per_class is not None:
        raise Warning(
            "The test_size_per_class will be considered if both test_size and test_size_per_class are given!"
        )
    if val_size is not None and val_size_per_class is not None:
        raise Warning("The val_size_per_class will be considered if both val_size and val_size_per_class are given!")
    if train_size is not None and train_size_per_class is not None:
        raise Warning(
            "The train_size_per_class will be considered if both train_size and val_size_per_class are given!"
        )

    if len(seed) == 0:
        seed = list(range(data_split))
    if len(seed) != data_split:
        raise ValueError("Please input the random seed list with the same length of {}!".format(data_split))

    if isinstance(data.y, torch.Tensor):
        labels = data.y.numpy()
    else:
        labels = np.array(data.y)
    masks = {}
    masks["train"], masks["val"], masks["test"], masks["seed"] = [], [], [], []
    for i in range(data_split):
        random_state = np.random.RandomState(seed[i])
        train_indices, val_indices, test_indices, seed_indices = get_train_val_test_seed_split(
            random_state,
            labels,
            train_size_per_class,
            val_size_per_class,
            test_size_per_class,
            seed_size_per_class,
            train_size,
            val_size,
            test_size,
            seed_size,
        )

        train_mask = np.zeros((labels.shape[0], 1), dtype=int)
        train_mask[train_indices, 0] = 1
        val_mask = np.zeros((labels.shape[0], 1), dtype=int)
        val_mask[val_indices, 0] = 1
        test_mask = np.zeros((labels.shape[0], 1), dtype=int)
        test_mask[test_indices, 0] = 1
        seed_mask = np.zeros((labels.shape[0], 1), dtype=int)
        if len(seed_indices) > 0:
            seed_mask[seed_indices, 0] = 1

        mask = {}
        mask["train"] = torch.from_numpy(train_mask).bool()
        mask["val"] = torch.from_numpy(val_mask).bool()
        mask["test"] = torch.from_numpy(test_mask).bool()
        mask["seed"] = torch.from_numpy(seed_mask).bool()

        masks["train"].append(mask["train"])
        masks["val"].append(mask["val"])
        masks["test"].append(mask["test"])
        masks["seed"].append(mask["seed"])

    data.train_mask = torch.cat(masks["train"], axis=-1)
    data.val_mask = torch.cat(masks["val"], axis=-1)
    data.test_mask = torch.cat(masks["test"], axis=-1)
    data.seed_mask = torch.cat(masks["seed"], axis=-1)
    return data


class Telegram(InMemoryDataset):
    r"""Data loader for the Telegram data set used in the
    `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        self.url = "https://github.com/SherylHYX/pytorch_geometric_signed_directed/raw/main/datasets/telegram"
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["telegram_adj.npz", "telegram_labels.npy"]

    @property
    def processed_file_names(self):
        return ["telegram.pt"]

    def download(self):
        for name in self.raw_file_names:
            download_url("{}/{}".format(self.url, name), self.raw_dir)

    def process(self):
        A = sp.load_npz(self.raw_paths[0])
        label = np.load(self.raw_paths[1])
        rs = np.random.RandomState(seed=0)

        test_ratio = 0.2
        train_ratio = 0.6
        val_ratio = 1 - train_ratio - test_ratio

        label = torch.from_numpy(label).long()
        s_A = sp.csr_matrix(A)
        coo = s_A.tocoo()
        values = coo.data

        indices = np.vstack((coo.row, coo.col))
        indices = torch.from_numpy(indices).long()
        features = torch.from_numpy(rs.normal(0, 1.0, (s_A.shape[0], 1))).float()

        data = Data(x=features, edge_index=indices, edge_weight=torch.FloatTensor(values), y=label)
        data = node_class_split(data, train_size_per_class=train_ratio, val_size_per_class=val_ratio)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
