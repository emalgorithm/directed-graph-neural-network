import os.path as osp
from typing import Callable, Optional
import gdown
import numpy as np

import torch
from torch_geometric.data import Data, InMemoryDataset


class DirectedHeterophilousGraphDataset(InMemoryDataset):
    r"""The directed heterophilous graphs :obj:`"Roman-empire"`,
    :obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"` and
    :obj:`"Questions"` from the `"A Critical Look at the Evaluation of GNNs
    under Heterophily: Are We Really Making Progress?"
    <https://arxiv.org/abs/2302.11640>`_ paper.
    """

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower().replace("-", "_")
        assert self.name in [
            "directed_roman_empire",
            "directed_amazon_ratings",
            "directed_questions",
        ]

        self.url = {
            "directed_roman_empire": "https://drive.google.com/uc?id=1atonwA1YqKMV3xWS7T04dRgfmDrsyRj8",
            "directed_amazon_ratings": "https://drive.google.com/uc?id=12Cyw0oZXLjPrebCficporBcIKiAgU5kc",
            "directed_questions": "https://drive.google.com/uc?id=1EnOvBehgLN3uCAQBXrGzGB1d3-aXS2Lk",
        }

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> str:
        return f"{self.name}.npz"

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def download(self):
        gdown.download(self.url[self.name], f"{self.raw_dir}/{self.name}.npz", fuzzy=True)

    def process(self):
        raw = np.load(self.raw_paths[0], "r")
        x = torch.from_numpy(raw["node_features"])
        y = torch.from_numpy(raw["node_labels"])
        edge_index = torch.from_numpy(raw["edges"]).t().contiguous()
        train_mask = torch.from_numpy(raw["train_masks"]).t().contiguous()
        val_mask = torch.from_numpy(raw["val_masks"]).t().contiguous()
        test_mask = torch.from_numpy(raw["test_masks"]).t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
