#%%
import torch
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data


#%%
edge_index: torch.Tensor = torch.tensor(
    [[0, 1, 1, 2], [1, 0, 2, 1]],
    dtype=torch.long
    )
x: torch.Tensor = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data: torch_geometric.data.data.Data = Data(x=x, edge_index=edge_index)


#%%
edge_index: torch.Tensor = torch.tensor(
    [[0, 1],
     [1, 0],
     [1, 2],
     [2, 1]], dtype=torch.long)

x: torch.Tensor = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data: torch_geometric.data.data.Data = Data(
    x=x, edge_index=edge_index.t().contiguous()
    )


#%%
print(data.keys)

print(data["x"])

for key, item in data: print(f"{key} found in data")

'edge_attr' in data

data.num_edges

data.num_node_features

data.has_isolated_nodes()

device: torch.device = torch.device("mps")\
    if torch.backends.mps.is_available() else torch.device("cpu")
data_on: torch_geometric.data.data.Data = data.to(device)

#%%

data_dir: str = "/Volumes/LaCie/data"

dataset: torch_geometric.datasets.tu_dataset.TUDataset\
    = TUDataset(root="/Volumes/LaCie/data/ENZYMES", name="ENZYMES")

len(dataset)

dataset.num_classes

dataset.num_node_features

data: torch_geometric.data.data.Data = dataset[0]
data.is_undirected()


