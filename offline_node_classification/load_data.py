from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
from torchvision import datasets, transforms
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import pdb

class load_cora_train:
    def __init__(self):
        # Fetch data
        batch_size = 1
        dataset = Planetoid(root='./data/Cora', name='Cora')
        self.data = dataset[0]

        # Extract training set
        self.train_mask = ~self.data.train_mask
        self.X_all = self.data.x[self.train_mask]
        self.edge_index_all = self.data.edge_index
        self.Y_all = self.data.y[self.train_mask]

        self.n_arm = 7
        self.node_size = self.X_all.shape[0]
        self.dim = 10031
        self.data_index = 0

    def step(self):  
        x_idx = self.data_index
        x, y = self.X_all[x_idx, :], self.Y_all[x_idx]
        d = x.numpy()
        target = int(y.item())
        X_n = []
        X_ind = [[x_idx, self.node_size + i] for i in range(self.n_arm)]
        for i in range(self.n_arm):
            front = np.zeros((1433 * i))
            back = np.zeros((1433 * (self.n_arm - 1 - i)))
            new_d = np.concatenate((front, d, back), axis=0)
            X_n.append(new_d)

        X_n = np.array(X_n)
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
        
        self.data_index += 1
        if self.data_index == len(self.X_all):
            self.data_index = 0
        node1_idx = x_idx
        node2_idx = self.node_size + int(y.item())
        return X_n, X_ind, rwd, target, node1_idx, node2_idx    
    
class load_cora_test:
    def __init__(self):
        # Fetch data
        batch_size = 1
        dataset = Planetoid(root='./data/Cora_', name='Cora')
        self.data = dataset[0]

        # Extract test set
        self.test_mask = self.data.train_mask
        self.X_all = self.data.x[self.test_mask]
        self.edge_index_all = self.data.edge_index
        self.Y_all = self.data.y[self.test_mask]

        self.n_arm = 7
        self.node_size = self.X_all.shape[0]
        self.dim = 10031
        self.data_index = 0

    def step(self):
        x_idx = self.data_index
        x, y = self.X_all[x_idx, :], self.Y_all[x_idx]
        d = x.numpy()
        target = int(y.item())
        X_n = []
        X_ind = [[x_idx, self.node_size + i] for i in range(self.n_arm)]
        for i in range(self.n_arm):
            front = np.zeros((1433 * i))
            back = np.zeros((1433 * (self.n_arm - 1 - i)))
            new_d = np.concatenate((front, d, back), axis=0)
            X_n.append(new_d)

        X_n = np.array(X_n)
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1

        self.data_index += 1
        if self.data_index == len(self.X_all):
            self.data_index = 0
        node1_idx = x_idx
        node2_idx = self.node_size + int(y.item())
        return X_n, X_ind, rwd, target, node1_idx, node2_idx      
    
    
class load_citeseer_train:
    def __init__(self):
        
        dataset = Planetoid(root='./data', name='CiteSeer')
        self.data = dataset[0]
        
        self.test_mask = self.data.test_mask
        self.valid_mask = self.data.val_mask
        self.train_mask = self.data.train_mask
        
        combined_mask = ~self.train_mask
        self.X_all = self.data.x[combined_mask]
        self.edge_index_all = self.data.edge_index
        self.Y_all = self.data.y[combined_mask]
        
        self.n_arm = 6
        self.node_size = 3327
        self.dim = 6 * 3703
        self.data_index = 0

    def step(self):  
        x_idx = self.data_index
        x, y = self.X_all[x_idx, :], self.Y_all[x_idx]
        d = x.numpy()

        target = int(y.item())
        # print(target)
        X_n = []
        X_ind = [[x_idx, self.node_size + i] for i in range(self.n_arm)]
        for i in range(self.n_arm):
            front = np.zeros((3703 * i))
            back = np.zeros((3703 * (5 - i)))
            new_d = np.concatenate((front, d, back), axis=0)
            X_n.append(new_d)

        X_n = np.array(X_n)
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1

        
        self.data_index += 1
        if self.data_index == len(self.X_all):
            self.data_index = 0
        node1_idx = x_idx
        node2_idx = self.node_size + int(y.item())
        # pdb.set_trace()
        return X_n, X_ind, rwd, target,node1_idx, node2_idx

class load_citeseer_test:
    def __init__(self):
        
        dataset = Planetoid(root='./data', name='CiteSeer')
        self.data = dataset[0]

        self.train_mask = self.data.train_mask
        self.X_all = self.data.x[self.train_mask]
        self.edge_index_all = self.data.edge_index
        self.Y_all = self.data.y[self.train_mask]

        self.n_arm = 6
        self.node_size = 3327
        self.dim = 6 * 3703
        self.data_index = 0

    def step(self):  
        x_idx = self.data_index
        x, y = self.X_all[x_idx, :], self.Y_all[x_idx]
        d = x.numpy()

        target = int(y.item())

        X_n = []
        X_ind = [[x_idx, self.node_size + i] for i in range(self.n_arm)]
        for i in range(self.n_arm):
            front = np.zeros((3703 * i))
            back = np.zeros((3703 * (5 - i)))
            new_d = np.concatenate((front, d, back), axis=0)
            X_n.append(new_d)

        X_n = np.array(X_n)
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
        
        self.data_index += 1
        if self.data_index == len(self.X_all):
            self.data_index = 0
        node1_idx = x_idx
        node2_idx = self.node_size + int(y.item())

        return X_n, X_ind, rwd, target,node1_idx, node2_idx    
    
class load_pubmed_train:
    def __init__(self):
        
        dataset = Planetoid(root='./data', name='PubMed')
        self.data = dataset[0]
        
        self.test_mask = self.data.test_mask
        self.valid_mask = self.data.val_mask
        self.train_mask = self.data.train_mask
        
        combined_mask = ~self.train_mask
        self.X_all = self.data.x[combined_mask]
        self.edge_index_all = self.data.edge_index
        self.Y_all = self.data.y[combined_mask]
        
        
        self.n_arm = 3
        self.node_size = 19717
        self.dim = 3 * 500
        self.data_index = 0

    def step(self):  
        x_idx = self.data_index
        x, y = self.X_all[x_idx, :], self.Y_all[x_idx]
        d = x.numpy()

        target = int(y.item())
        # print(target)
        X_n = []
        X_ind = [[x_idx, self.node_size + i] for i in range(self.n_arm)]
        for i in range(self.n_arm):
            front = np.zeros((500 * i))
            back = np.zeros((500 * (2 - i)))
            new_d = np.concatenate((front, d, back), axis=0)
            X_n.append(new_d)

        X_n = np.array(X_n)
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
        
        
        self.data_index += 1
        if self.data_index == 19717:
            self.data_index = 0
        node1_idx = x_idx
        node2_idx = self.node_size + int(y.item())
        # pdb.set_trace()
        return X_n, X_ind, rwd, target,node1_idx, node2_idx

class load_pubmed_test:
    def __init__(self):
        
        dataset = Planetoid(root='./data', name='PubMed')
        self.data = dataset[0]
        
        self.train_mask = self.data.train_mask
        self.X_all = self.data.x[self.train_mask]
        self.edge_index_all = self.data.edge_index
        self.Y_all = self.data.y[self.train_mask]

        self.n_arm = 3
        self.node_size = 19717
        self.dim = 3 * 500
        self.data_index = 0

    def step(self):  
        x_idx = self.data_index
        x, y = self.X_all[x_idx, :], self.Y_all[x_idx]
        d = x.numpy()

        target = int(y.item())

        X_n = []
        X_ind = [[x_idx, self.node_size + i] for i in range(self.n_arm)]
        for i in range(self.n_arm):
            front = np.zeros((500 * i))
            back = np.zeros((500 * (2 - i)))
            new_d = np.concatenate((front, d, back), axis=0)
            X_n.append(new_d)

        X_n = np.array(X_n)
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
         
        self.data_index += 1
        if self.data_index == 19717:
            self.data_index = 0
        node1_idx = x_idx
        node2_idx = self.node_size + int(y.item())
        # pdb.set_trace()
        return X_n, X_ind, rwd, target,node1_idx, node2_idx   
    
if __name__ == '__main__':
    dataset = load_cora_train()
    dataset.step()
