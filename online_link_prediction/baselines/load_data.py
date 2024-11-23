from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd 
import scipy.sparse as sp
import torch
import torchvision
from torchvision import datasets, transforms
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import pdb

class load_cora:
    def __init__(self):
        batch_size = 1
        dataset = Planetoid(root='./data/Cora_', name='Cora')
        self.data = dataset[0]
        
        self.X_all,self.edge_index_all, self.Y_all = self.data.x, self.data.edge_index, self.data.y
        self.n_arm = 7
        self.node_size = 2708
        self.dim = 10031
        self.data_index = 0

    def step(self):  
        x_idx = self.data_index
        x, y = self.X_all[x_idx, :], self.Y_all[x_idx]
        d = x.numpy()
        # print(d.shape)
        # d = d.reshape(1433)
        # print(d.shape)
        target = int(y.item())
        # print(target)
        X_n = []
        X_ind = [[x_idx, self.node_size + i] for i in range(self.n_arm)]
        for i in range(7):
            front = np.zeros((1433 * i))
            back = np.zeros((1433 * (6 - i)))
            new_d = np.concatenate((front, d, back), axis=0)
            X_n.append(new_d)

        X_n = np.array(X_n)
        rwd = np.zeros(self.n_arm)
        rwd[target] = 1
        
        self.data_index += 1
        if self.data_index == 2708:
            self.data_index = 0
        node1_idx = x_idx
        node2_idx = self.node_size + int(y.item())
        # pdb.set_trace()
        return X_n, X_ind, rwd, target,node1_idx, node2_idx

class load_citeseer:
    def __init__(self):
        
        dataset = Planetoid(root='./data', name='CiteSeer')
        self.data = dataset[0]
        
        self.X_all,self.edge_index_all, self.Y_all = self.data.x, self.data.edge_index, self.data.y
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
        if self.data_index == 3327:
            self.data_index = 0
        node1_idx = x_idx
        node2_idx = self.node_size + int(y.item())
        # pdb.set_trace()
        return X_n, X_ind, rwd, target,node1_idx, node2_idx
    
class load_pubmed:
    def __init__(self):
        
        dataset = Planetoid(root='./data', name='PubMed')
        self.data = dataset[0]
        
        self.X_all,self.edge_index_all, self.Y_all = self.data.x, self.data.edge_index, self.data.y
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
    
class load_movielen:
    def __init__(self):
        # Fetch data
        self.m = np.load("./data/MovieLens/movie_2000users_10000items_entry.npy")
        self.U = np.load("./data/MovieLens/movie_2000users_10000items_features.npy")
        self.I = np.load("./data/MovieLens/movie_10000items_2000users_features.npy")
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else: # i[2] == -1
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        # print('self.p_d and self.n_d is:',self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), 9, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), replace=False)]
        X_ind = np.concatenate((pos[:arm], [neg], pos[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]]))) 
            if  i == arm:
                user = ind[0]
                item = ind[1]
        # print("X is \n",X)
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        # pdb.set_trace()
        return np.array(X),X_ind, rwd, arm, user, item  # arm is the one that randomly picked up and settled to 1
    
    

class load_facebook:
    def __init__(self):
        # Fetch data
        self.m = np.load("./data/Facebook/facebook_combined_ALLusers_entry.npy")
        self.U = np.load("./data/Facebook/facebook_combined_ALLusers_features.npy")
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else: # i[2] == -1
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        # print('self.p_d and self.n_d is:',self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), 9, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.U[ind[1]]))) 
            if arm == 9 and i == arm:
                user = ind[0]
                item = ind[1]
            elif i == arm + 1 : 
                user = ind[0]
                item = ind[1]
        # print("X is \n",X)
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X),X_ind, rwd, arm, user, item  # arm is the one that randomly picked up and settled to 1


class load_grqc:
    def __init__(self):
        # Fetch data
        self.m = np.load("./data/GrQc/GrQc_ALLusers_entry.npy")
        self.U = np.load("./data/GrQc/GrQc_ALLusers_features.npy")
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else: # i[2] == -1
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        # print('self.p_d and self.n_d is:',self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), 9, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.U[ind[1]]))) 
            if arm == 9 and i == arm:
                user = ind[0]
                item = ind[1]
            elif i == arm + 1 : 
                user = ind[0]
                item = ind[1]
        # print("X is \n",X)
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X),X_ind, rwd, arm, user, item  # arm is the one that randomly picked up and settled to 1

class load_amazon_fashion:
    def __init__(self):
        # Fetch data
        self.m = np.load("./data/Amazon_fashion/new/amazon_fashion_4000users_entry.npy")
        self.U = np.load("./data/Amazon_fashion/new/amazon_fashion_4000users_4000items_features.npy")
        self.I = np.load("./data/Amazon_fashion/new/amazon_fashion_4000items_4000users_features.npy")
        self.n_arm = 10
        self.dim = 20
        self.pos_index = []
        self.neg_index = []
        for i in self.m:
            if i[2] ==1:
                self.pos_index.append((i[0], i[1]))
            else:
                self.neg_index.append((i[0], i[1]))   
            
        self.p_d = len(self.pos_index)
        self.n_d = len(self.neg_index)
        print(self.p_d, self.n_d)
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10))
        #print(pos_index.shape)
        pos = self.pos_index[np.random.choice(range(self.p_d), replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), 9, replace=False)]
        X_ind = np.concatenate((neg[:arm], [pos], neg[arm:]), axis=0) 
        # print("X_ind is:",X_ind)
        X = []
        for i,ind in enumerate(X_ind):
            #X.append(np.sqrt(np.multiply(self.I[ind], u_fea)))
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]]))) 
            if arm == 9 and i == arm:
                user = ind[0]
                item = ind[1]
            elif i == arm + 1 : 
                user = ind[0]
                item = ind[1]
        rwd = np.zeros(self.n_arm)
        rwd[arm] = 1
        return np.array(X),X_ind, rwd, arm, user, item  # arm is the one that randomly picked up and settled to 1
    
