from torch.utils.data import Dataset


class FullBatchGraphDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        # There is only one sample which is the whole graph
        return 1

    def __getitem__(self, idx):
        return self.data
