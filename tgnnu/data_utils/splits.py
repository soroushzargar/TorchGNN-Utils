from abc import ABC
import torch
import math

import torch_geometric
from torch_geometric.data import Data as GraphData

# Defining Split Manager
class SplitManager(ABC):
    def __init__(self, dataset):
        """
        Initializes the Splits object. The object is used to sample masks iteratively for training, validation and testing.

        Args:
            dataset (torch_geometric.data.Data): The input dataset.

        Attributes:
            device (torch.device): The device on which the data is stored.
            dataset (torch_geometric.data.Data): The input dataset.
            perm_idx (torch.Tensor): The randomly permuted indices of the dataset. The iterative steps will be based on these indices.
            perm_selected (torch.Tensor): A boolean tensor indicating which indices are selected.
            perm_class (torch.Tensor): The class labels corresponding to the permuted indices.
        """
        self.device = dataset.x.device
        self.dataset = dataset
        self.perm_idx = torch.randperm(self.dataset.x.shape[0]).to(self.device)
        self.perm_selected = torch.zeros_like(self.perm_idx).bool().to(self.device)
        self.perm_class = self.dataset.y[self.perm_idx]

    def alloc(self, budget, budget_allocated="overall", stratified=False, return_cumulative=False, return_mask=True):
        """ Allocates a set of indices based on the budget and the budget allocation strategy
        """
        if budget_allocated == "overall" and stratified:
            budget = math.ceil(budget / len(torch.unique(self.dataset.y)))

        if budget_allocated == "per_class" and stratified == False:
            raise ValueError("Budget allocation per class is only possible with stratified sampling")

        if stratified == False:
            selected = self.perm_idx[~self.perm_selected][:budget]
            flipping_idx = (self.perm_selected == False).nonzero(as_tuple=True)[0][:budget]
            self.perm_selected[flipping_idx] = True

        else:
            overall_selected = []
            for class_idx in torch.unique(self.perm_class):
                cls_idx = class_idx.item()
                class_pidx = self.perm_idx[(~self.perm_selected) & (self.perm_class == cls_idx)]
                class_selected = class_pidx[:min(budget, class_pidx.shape[0])]
                overall_selected.append(class_selected)
            overall_selected = torch.concat(overall_selected)
            out = torch.zeros_like(self.perm_idx).bool()
            out[overall_selected] = True
            self.perm_selected = self.perm_selected | out[self.perm_idx]
            selected = overall_selected


        if return_cumulative:
            result = self.perm_idx[self.perm_selected]
        else:
            result = selected

        if return_mask == True:
            out = torch.zeros_like(self.perm_idx).bool()
            out[result] = True
            return out
        else:
            out = result
            return out
        
    def shuffle_free_idxs(self):
        free_idxs = self.perm_idx[~self.perm_selected]
        new_perm_unselected = torch.randperm(free_idxs.shape[0])

        # updaing perm_idx
        self.perm_idx[~self.perm_selected] = free_idxs[new_perm_unselected]

        # updating perm_class
        free_classes = self.perm_class[~self.perm_selected]
        self.perm_class[~self.perm_selected] = free_classes[new_perm_unselected]
        


class GraphSplit(object):
    def __init__(self, n_vertices, n_edges, edge_index, ys=None, device='cpu', undirected=True):
        """
        Initializes the Splitter object.

        Args:
            n_vertices (int): The number of vertices in the graph.
            n_edges (int): The number of edges in the graph.
            edge_index (Tensor): The edge index tensor representing the graph structure.
            ys (Tensor, optional): The target labels for each vertex. Defaults to None.
            device (str, optional): The device to store the tensors on. Defaults to 'cpu'.
            undirected (bool, optional): Whether the graph is undirected. Defaults to True.
        """
        self.device = device
        self.n_vertices = n_vertices
        self.n_edges = n_edges // 2 if undirected else n_edges

        self._vertices_budget = torch.ones(size=(self.n_vertices,), dtype=bool).to(self.device)
        self._edge_budget = torch.ones(size=(self.n_edges, ), dtype=bool).to(self.device)
        self.ys = ys
        self.stratified_enabled = False
        self._n_classes = None
        if not self.ys is None:
            self.stratified_enabled = True
            self._n_classes = ys.max() + 1
        if undirected:
            self.edge_index = self.onedir_edge_index(edge_index)
        else:
            self.edge_index = edge_index

    @staticmethod
    def onedir_edge_index(edge_index):
        """
        Filters the given edge_index to handle undirected graphs.
        
        Args:
            edge_index (torch.Tensor): The edge index tensor of shape (2, num_edges) representing the graph edges.
        
        Returns:
            torch.Tensor: The filtered edge index tensor of shape (2, num_filtered_edges) where the source node index is less than the target node index.
        """
        e_filter = edge_index[0] < edge_index[1]
        return edge_index.T[e_filter].T

    def sample_nodes(self, n_nodes, return_mask=True, stratified=False):
        if stratified and (self.stratified_enabled == False):
            raise KeyError("Stratified sampling is not supported if n_classes is not mentioend")
        node_idxs = None
        if stratified:
            node_idxs = self._stratified_sample_nodes(n_nodes=n_nodes)
        else:
            node_idxs = self._overal_sample_nodes(n_nodes=n_nodes)

        node_mask = self._convert_to_mask(node_idxs, self.n_vertices)
        corresponding_edge_mask = self._compute_subgraph_edges(node_mask)
        self._vertices_budget = self._vertices_budget & (~node_mask)
        self._edge_budget = self._edge_budget & (~corresponding_edge_mask)

        if return_mask:
            return node_mask
        else:
            return node_mask.nonzero(as_tuple=True)[0]
    
    def sample_edges(self, n_edges, return_mask=False):
        edge_idxs = self._overal_sample_edges(n_edges=n_edges)
        edge_mask = self._convert_to_mask(edge_idxs, self.n_edges)

        # TODO: extract from nodes and edges
        used_vertices = torch.zeros_like(self._vertices_budget).to(self.device)
        used_vertices[self.edge_index_filter(self.edge_index, edge_mask).reshape(-1, )] = True

        self._vertices_budget = self._vertices_budget & (~used_vertices)
        self._edge_budget = self._edge_budget & (~edge_mask)

        if return_mask:
            return edge_mask
        else:
            return self.return_undirected_edges(edge_mask)
    
    def return_undirected_edges(self, edge_mask):
        e = self.edge_index_filter(self.edge_index, edge_mask)
        return torch_geometric.utils.to_undirected(e)
        
    @staticmethod
    def edge_index_filter(edge_index, edge_mask):
        return edge_index.T[edge_mask].T

    def compute_node_mask(self, edge_mask):
        used_vertices = torch.zeros_like(self._vertices_budget).to(self.device)
        used_vertices[self.edge_index_filter(self.edge_index, edge_mask).reshape(-1, )] = True
        return used_vertices
        
    def _overal_sample_edges(self, n_edges):
        target_edge_mask = self._edge_budget
        idxs = target_edge_mask.nonzero(as_tuple=True)[0][torch.randperm(target_edge_mask.sum())]
        return idxs[:n_edges]


    def _convert_to_mask(self, idx, n_elems):
        res = torch.zeros((n_elems, ), dtype=bool).to(self.device)
        res[idx] = True
        return res

    def _compute_subgraph_edges(self, node_mask):
        edge_mask = node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]
        return edge_mask

        
    def _overal_sample_nodes(self, n_nodes):
        target_node_mask = self._vertices_budget
        idxs = target_node_mask.nonzero(as_tuple=True)[0][torch.randperm(target_node_mask.sum()).to(self.device)]
        return idxs[:n_nodes]
        
        
    def _stratified_sample_nodes(self, n_nodes):
        result_idxs = []
        for class_i in range(self._n_classes):
            target_node_mask = ((self.ys == class_i) & (self._vertices_budget))
            idxs = target_node_mask.nonzero(as_tuple=True)[0][torch.randperm(target_node_mask.sum()).to(self.device)]
            result_idxs.append(idxs[:n_nodes])
        return torch.concat(result_idxs)

    @classmethod
    def from_dataset(cls, dataset):
        obj = cls(n_vertices=dataset.x.shape[0], n_edges=dataset.edge_index.shape[1], edge_index=dataset.edge_index, ys=dataset.y, device=dataset.x.device)
        # obj.device = dataset.x.device
        return obj

# node-induced subgraph
def node_induced_subgraph(graph, mask):
    new_edge_index = graph.edge_index.T[
        mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
        ].T.clone()
    
    return GraphData(x=graph.x, edge_index=new_edge_index, y=graph.y)


# edge-induced subgraph
def edge_induced_subgraph(graph, edge_mask):
    new_edge_index = graph.edge_index.T[edge_mask].T.clone()
    return GraphData(x=graph.x, edge_index=new_edge_index, y=graph.y)

# Union of two edges
def union_edge_index(graph, first, second):
    edge_index = torch_geometric.utils.sort_edge_index(torch.concat([first, second], dim=1))
    return GraphData(x=graph.x, edge_index=edge_index, y=graph.y)