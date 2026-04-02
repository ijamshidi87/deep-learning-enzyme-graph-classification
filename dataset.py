import os
import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data, DataLoader


# Load ENZYMES dataset from raw text files
def load_enzymes_raw(data_dir='ENZYMES'):
    # Read all files (comma-separated)
    graph_indicator = np.loadtxt(os.path.join(data_dir, 'ENZYMES_graph_indicator.txt'), dtype=int, delimiter=',')
    node_labels = np.loadtxt(os.path.join(data_dir, 'ENZYMES_node_labels.txt'), dtype=int, delimiter=',')
    edges = np.loadtxt(os.path.join(data_dir, 'ENZYMES_A.txt'), dtype=int, delimiter=',')
    node_attributes = np.loadtxt(os.path.join(data_dir, 'ENZYMES_node_attributes.txt'), dtype=float, delimiter=',')
    graph_labels = np.loadtxt(os.path.join(data_dir, 'ENZYMES_graph_labels.txt'), dtype=int, delimiter=',')
    
    # Adjust to zero-based indexing
    graph_indicator = graph_indicator - 1
    node_labels = node_labels - 1
    edges = edges - 1
    graph_labels = graph_labels - 1
    
    # Get number of graphs
    num_graphs = len(graph_labels)
    
    # Create graph list
    graphs = []
    
    for graph_id in range(num_graphs):
        # Get nodes for this graph
        node_mask = (graph_indicator == graph_id)
        node_indices = np.where(node_mask)[0]
        
        # Create node index mapping
        node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Get edges for this graph
        edge_mask = np.isin(edges[:, 0], node_indices) & np.isin(edges[:, 1], node_indices)
        graph_edges = edges[edge_mask]
        
        # Remap edge indices
        graph_edges_remapped = np.array([[node_map[e[0]], node_map[e[1]]] for e in graph_edges])
        
        # Convert to undirected (add reverse edges)
        edge_index = np.concatenate([graph_edges_remapped, graph_edges_remapped[:, [1, 0]]], axis=0)
        edge_index = torch.tensor(edge_index.T, dtype=torch.long)
        edge_index = torch.unique(edge_index, dim=1)
        
        # Get node features (combine attributes and labels)
        node_feat = np.hstack([node_attributes[node_indices], node_labels[node_indices].reshape(-1, 1)])
        x = torch.tensor(node_feat, dtype=torch.float)
        
        # Get graph label
        y = torch.tensor([graph_labels[graph_id]], dtype=torch.long)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = x.shape[0]
        graphs.append(data)
    
    return graphs


# Split dataset into train/val/test
def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    torch.manual_seed(seed)
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset, test_dataset


# Create DataLoaders for train/val/test
def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# Get dataset statistics
def get_dataset_info(dataset):
    num_graphs = len(dataset)
    num_classes = len(set([data.y.item() for data in dataset]))
    num_features = dataset[0].x.shape[1]
    
    avg_nodes = np.mean([data.x.shape[0] for data in dataset])
    avg_edges = np.mean([data.edge_index.shape[1] for data in dataset]) / 2
    
    info = {
        'num_graphs': num_graphs,
        'num_classes': num_classes,
        'num_features': num_features,
        'avg_nodes': avg_nodes,
        'avg_edges': avg_edges
    }
    
    return info
