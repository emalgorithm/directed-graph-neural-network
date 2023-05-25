import numpy as np
import scipy
import os
import gdown

import torch
from torch_geometric.data import Data
from src.datasets.data_utils import get_mask


# Taken verbatim from https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data_utils.py#L39
def even_quantile_labels(vals, nclasses, verbose=True):
    """partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
    """
    label = -1 * np.ones(vals.shape[0], dtype=np.int64)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print("Class Label Intervals:")
        for class_idx, interval in enumerate(interval_lst):
            print(f"Class {class_idx}: [{interval[0]}, {interval[1]})]")
    return label


# adapting
# https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data_utils.py#L221
# load splits from here https://github.com/CUAI/Non-Homophily-Large-Scale/tree/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/data/splits
def process_fixed_splits(splits_lst, num_nodes):
    n_splits = len(splits_lst)
    train_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, n_splits, dtype=torch.bool)
    for i in range(n_splits):
        train_mask[splits_lst[i]["train"], i] = 1
        val_mask[splits_lst[i]["valid"], i] = 1
        test_mask[splits_lst[i]["test"], i] = 1
    return train_mask, val_mask, test_mask


# adapting - https://github.com/CUAI/Non-Homophily-Large-Scale/blob/82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6/dataset.py#L257
def load_snap_patents_mat(n_classes=5, root="dataset/"):
    dataset_drive_url = {"snap-patents": "1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia"}
    splits_drive_url = {"snap-patents": "12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N"}

    # Build dataset folder
    if not os.path.exists(f"{root}snap_patents"):
        os.mkdir(f"{root}snap_patents")

    # Download the data
    if not os.path.exists(f"{root}snap_patents/snap_patents.mat"):
        p = dataset_drive_url["snap-patents"]
        print(f"Snap patents url: {p}")
        gdown.download(
            id=dataset_drive_url["snap-patents"],
            output=f"{root}snap_patents/snap_patents.mat",
            quiet=False,
        )

    # Get data
    fulldata = scipy.io.loadmat(f"{root}snap_patents/snap_patents.mat")
    edge_index = torch.tensor(fulldata["edge_index"], dtype=torch.long)
    node_feat = torch.tensor(fulldata["node_feat"].todense(), dtype=torch.float)
    num_nodes = int(fulldata["num_nodes"])
    years = fulldata["years"].flatten()
    label = even_quantile_labels(years, n_classes, verbose=False)
    label = torch.tensor(label, dtype=torch.long)

    # Download splits
    name = "snap-patents"
    if not os.path.exists(f"{root}snap_patents/{name}-splits.npy"):
        assert name in splits_drive_url.keys()
        gdown.download(
            id=splits_drive_url[name],
            output=f"{root}snap_patents/{name}-splits.npy",
            quiet=False,
        )

    # Get splits
    splits_lst = np.load(f"{root}snap_patents/{name}-splits.npy", allow_pickle=True)
    train_mask, val_mask, test_mask = process_fixed_splits(splits_lst, num_nodes)
    data = Data(
        x=node_feat,
        edge_index=edge_index,
        y=label,
        num_nodes=num_nodes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    dataset = DummyDataset(data, n_classes)

    return dataset


class DummyDataset(object):
    def __init__(self, data, num_classes):
        self.data = data
        self.num_classes = num_classes
