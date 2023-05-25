import os
import pickle

import torch_scatter
import torch
import numpy as np
from sklearn.preprocessing import normalize
from networkx import DiGraph
from torch_geometric.utils import from_networkx
from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.data import Data, InMemoryDataset

def get_syn_dataset(path):
    dataset_path = os.path.join(path, "syn-dir", "processed", "dataset.pkl")
    if os.path.isfile(dataset_path):
        dataset = pickle.load(open(dataset_path, "rb"))
    else:
        os.makedirs(os.path.join(path, "syn-dir", "processed"), exist_ok=True)
        #WikipediaNetwork(root=path, name="chameleon")
        num_nodes = 5000
        edge_index = erdos_renyi_graph(num_nodes=num_nodes, edge_prob=0.001, directed=True)
        row, col = dataset._data.edge_index
        
        # Generate random numbers between -1 and 1
        x = torch.rand(num_nodes) * 2 - 1
        x_mean_in = torch_scatter.scatter_mean(x[col], row, dim_size=num_nodes)
        x_mean_out = torch_scatter.scatter_mean(x[row], col, dim_size=num_nodes)
        # dataset._data.y = torch.round(torch.rand(num_nodes)).long().unsqueeze(-1)
        y = (x_mean_in > x_mean_out).long().unsqueeze(-1)
        
        data = Data(x=x.unsqueeze(dim=-1), y=y, edge_index=edge_index)
        dataset = InMemoryDataset()
        dataset._data = data
        pickle.dump(dataset, open(dataset_path, "wb"))

    return dataset


def generate_synthetic_directed_pa_graph(num_classes=5, num_nodes=1000, m=2, h=0.1):
    # Get compatibility matrix with given homophily
    H = np.random.rand(num_classes, num_classes)
    np.fill_diagonal(H, 0)
    H = (1 - h) * normalize(H, axis=1, norm='l1') 
    np.fill_diagonal(H, h)
    np.testing.assert_allclose(H.sum(axis=1), np.ones(num_classes) ,rtol=1e-5, atol=0)

    # Generate graph
    G = DiGraph()
    y = []
    for u in range(num_nodes):
        G.add_node(u)
        y_u = np.random.choice(range(num_classes))
        y.append(y_u)

        # Get probabilities for neighbors, proporational to in_degree and compatibility
        scores = np.array([(G.in_degree(v) + 0.01) * H[y[u], y[v]] for v in G])
        scores /= scores.sum()
        
        # Sample (at most) m neighbors according to the above probabilities
        num_edges = m if m <= G.number_of_nodes() else G.number_of_nodes()
        vs = np.random.choice(G.nodes(), size=num_edges, replace=False, p=scores)
        G.add_edges_from([(u, v) for v in vs])

    data = from_networkx(G)
    data.y = torch.Tensor(y)

    return data