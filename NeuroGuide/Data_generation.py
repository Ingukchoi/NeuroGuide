import torch
from torch.utils.data import Dataset 
from sklearn.cluster import KMeans
from torch_geometric.data import Data
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TSPDataset(Dataset):
    def __init__(self, n_nodes, n_samples, n_cluster, seed=None):
        super(TSPDataset, self).__init__()
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.n_cluster = n_cluster
        self.seed = seed
        self.set_seed(seed)
        self.data = self.make_dataset()

    def set_seed(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def make_dataset(self):
        self.set_seed(self.seed)

        dataset = torch.rand(self.n_samples, self.n_nodes, 3)
        augmented_dataset = torch.zeros(self.n_samples, self.n_nodes, 17)

        for i in range(self.n_samples):
            x, y = dataset[i, :, 0], dataset[i, :, 1]
            
            # feature augmentation
            dat1 = torch.stack((x, y), dim=1)
            dat2 = torch.stack((1 - x, y), dim=1)
            dat3 = torch.stack((x, 1 - y), dim=1)
            dat4 = torch.stack((1 - x, 1 - y), dim=1)
            dat5 = torch.stack((y, x), dim=1)
            dat6 = torch.stack((1 - y, x), dim=1)
            dat7 = torch.stack((y, 1 - x), dim=1)
            dat8 = torch.stack((1 - y, 1 - x), dim=1)
            
            augmented_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=1)
            augmented_dataset[i, :, :16] = augmented_data
            
            data_for_clustering = dataset[i, 1:, :2].cpu().numpy()
            kmeans = KMeans(n_clusters=self.n_cluster, init='k-means++', max_iter=100, random_state=self.seed, n_init=10).fit(data_for_clustering)
            cluster_labels = kmeans.labels_ + 1
            
            augmented_dataset[i, 0, 16] = 0
            augmented_dataset[i, 1:, 16] = torch.from_numpy(cluster_labels).cuda()

        return augmented_dataset

class TSPDataset_wo_aug(Dataset):
    def __init__(self, n_nodes, n_samples, n_cluster, seed=None):
        super(TSPDataset_wo_aug, self).__init__()
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.n_cluster = n_cluster
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self.data = self.make_dataset()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def make_dataset(self):
      dataset = torch.rand(self.n_samples, self.n_nodes, 3)

      for i in range(self.n_samples):
        data_for_clustering = dataset[i, 1:, :2].reshape(-1, 2).numpy()
        kmeans = KMeans(n_clusters= self.n_cluster, init='k-means++', max_iter=100, random_state=self.seed , n_init = 10).fit(data_for_clustering)
        cluster_labels = kmeans.labels_+1
        
        dataset[i, 0, 2] = 0
        dataset[i, 1:, 2] = torch.from_numpy(cluster_labels)
      return dataset
    

def create_graph_data(pos_input):
    batch_size, num_node, _ = pos_input.shape
    device = pos_input.device
    node_indices = torch.arange(num_node, device=device)
    edge_index = torch.combinations(node_indices, r=2).T
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    

    cluster_info = pos_input[:, :, -1]
    edge_attr = torch.abs(cluster_info[:, edge_index[0]] - cluster_info[:, edge_index[1]]) < 0.5
    edge_attr = edge_attr.long()

    cluster_edge_mask = edge_attr == 1

    data_list = []
    cluster_data_list = []

    for i in range(batch_size):
        data = Data(x=pos_input[i, :, :16], 
                    edge_index=edge_index,
                    edge_attr=edge_attr[i].unsqueeze(1))

        cluster_edge_index = edge_index[:, cluster_edge_mask[i]]
        cluster_data = Data(x=pos_input[i, :, :16], 
                            edge_index=cluster_edge_index)

        data_list.append(data)
        cluster_data_list.append(cluster_data)
    return data_list, cluster_data_list 

