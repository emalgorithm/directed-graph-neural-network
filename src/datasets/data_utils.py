import torch
from torch_sparse import mul
from torch_sparse import sum as sparsesum
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric
from torch_sparse import SparseTensor
import torch


def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))


def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj


def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")


def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def get_adj(edge_index, num_nodes, graph_type="directed"):
    """
    Return the type of adjacency matrix specified by `graph_type` as sparse tensor.
    """
    if graph_type == "transpose":
        edge_index = torch.stack([edge_index[1], edge_index[0]])
    elif graph_type == "undirected":
        edge_index = torch_geometric.utils.to_undirected(edge_index)
    elif graph_type == "directed":
        pass
    else:
        raise ValueError(f"{graph_type} is not a valid graph type")

    value = torch.ones((edge_index.size(1),), device=edge_index.device)
    return SparseTensor(row=edge_index[0], col=edge_index[1], value=value, sparse_sizes=(num_nodes, num_nodes))


def compute_unidirectional_edges_ratio(edge_index):
    num_directed_edges = edge_index.shape[1]
    num_undirected_edges = torch_geometric.utils.to_undirected(edge_index).shape[1]

    num_unidirectional = num_undirected_edges - num_directed_edges

    return (num_unidirectional / (num_undirected_edges / 2)) * 100
