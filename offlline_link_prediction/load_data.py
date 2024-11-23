from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np

class load_collab_train:
    def __init__(self):
        # Fetch data
        m = np.load("./data/collab/collab_users_entry_train.npy")
        self.m = [arr.astype(int) for arr in m]
        self.U = np.load("./data/collab/collab_users_items_features.npy")
        self.I = np.load("./data/collab/collab_items_users_features.npy")
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
        arm = np.random.choice(range(10),5,replace=False)
        pos = self.pos_index[np.random.choice(range(self.p_d), 5, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), 5, replace=False)]
        X_ind = np.zeros((10,2), dtype=int)
        X_ind[arm] = pos
        non_arm = [i for i in range(10) if i not in arm]
        X_ind[non_arm] = neg
        
        X = []
        for i,ind in enumerate(X_ind):
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]]))) 
            if i in arm:
                user = ind[0]
                item = ind[1]
        rwd = np.zeros(self.n_arm,dtype=int)
        rwd[arm] = 1
        return np.array(X),X_ind, rwd, arm, user, item 

class load_collab_test:
    def __init__(self):
        # Fetch data
        m = np.load("./data/collab/collab_users_entry_test.npy")
        self.m = [arr.astype(int) for arr in m]
        self.U = np.load("./data/collab/collab_users_items_features.npy")
        self.I = np.load("./data/collab/collab_items_users_features.npy")
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
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10),5,replace=False)
        pos = self.pos_index[np.random.choice(range(self.p_d), 5, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), 5, replace=False)]
        X_ind = np.zeros((10,2), dtype=int)
        X_ind[arm] = pos
        non_arm = [i for i in range(10) if i not in arm]
        X_ind[non_arm] = neg
        
        X = []
        for i,ind in enumerate(X_ind):
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]]))) 

            if i in arm:
                user = ind[0]
                item = ind[1]
        rwd = np.zeros(self.n_arm,dtype=int)
        rwd[arm] = 1

        return np.array(X),X_ind, rwd, arm, user, item 

class load_ppa_train:
    def __init__(self):
        # Fetch data
        m = np.load("./data/ppa/ppa_users_entry_train.npy")
        self.m = [arr.astype(int) for arr in m]
        self.U = np.load("./data/ppa/ppa_users_items_features.npy")
        self.I = np.load("./data/ppa/ppa_items_users_features.npy")
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
        arm = np.random.choice(range(10),5,replace=False)
        pos = self.pos_index[np.random.choice(range(self.p_d), 5, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), 5, replace=False)]
        X_ind = np.zeros((10,2), dtype=int)
        X_ind[arm] = pos
        non_arm = [i for i in range(10) if i not in arm]
        X_ind[non_arm] = neg
        
        X = []
        for i,ind in enumerate(X_ind):
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]]))) 
            if i in arm:
                user = ind[0]
                item = ind[1]
        rwd = np.zeros(self.n_arm,dtype=int)
        rwd[arm] = 1
        return np.array(X),X_ind, rwd, arm, user, item 

class load_ppa_test:
    def __init__(self):
        # Fetch data
        m = np.load("./data/ppa/ppa_users_entry_test.npy")
        self.m = [arr.astype(int) for arr in m]
        self.U = np.load("./data/ppa/ppa_users_items_features.npy")
        self.I = np.load("./data/ppa/ppa_items_users_features.npy")
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
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10),5,replace=False)
        pos = self.pos_index[np.random.choice(range(self.p_d), 5, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), 5, replace=False)]
        X_ind = np.zeros((10,2), dtype=int)
        X_ind[arm] = pos
        non_arm = [i for i in range(10) if i not in arm]
        X_ind[non_arm] = neg
        
        X = []
        for i,ind in enumerate(X_ind):
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]]))) 

            if i in arm:
                user = ind[0]
                item = ind[1]
        rwd = np.zeros(self.n_arm,dtype=int)
        rwd[arm] = 1

        return np.array(X),X_ind, rwd, arm, user, item 
    
class load_ddi_train:
    def __init__(self):
        # Fetch data
        m = np.load("./data/ddi/ddi_users_entry_train.npy")
        self.m = [arr.astype(int) for arr in m]
        self.U = np.load("./data/ddi/ddi_users_items_features.npy")
        self.I = np.load("./data/ddi/ddi_items_users_features.npy")
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
        arm = np.random.choice(range(10),5,replace=False)
        pos = self.pos_index[np.random.choice(range(self.p_d), 5, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), 5, replace=False)]
        X_ind = np.zeros((10,2), dtype=int)
        X_ind[arm] = pos
        non_arm = [i for i in range(10) if i not in arm]
        X_ind[non_arm] = neg
        
        X = []
        for i,ind in enumerate(X_ind):
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]]))) 
            if i in arm:
                user = ind[0]
                item = ind[1]
        rwd = np.zeros(self.n_arm,dtype=int)
        rwd[arm] = 1
        return np.array(X),X_ind, rwd, arm, user, item 

class load_ddi_test:
    def __init__(self):
        # Fetch data
        m = np.load("./data/ddi/ddi_users_entry_test.npy")
        self.m = [arr.astype(int) for arr in m]
        self.U = np.load("./data/ddi/ddi_users_items_features.npy")
        self.I = np.load("./data/ddi/ddi_items_users_features.npy")
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
        self.pos_index = np.array(self.pos_index)
        self.neg_index = np.array(self.neg_index)


    def step(self):        
        arm = np.random.choice(range(10),5,replace=False)
        pos = self.pos_index[np.random.choice(range(self.p_d), 5, replace=False)]
        neg = self.neg_index[np.random.choice(range(self.n_d), 5, replace=False)]
        X_ind = np.zeros((10,2), dtype=int)
        X_ind[arm] = pos
        non_arm = [i for i in range(10) if i not in arm]
        X_ind[non_arm] = neg
        
        X = []
        for i,ind in enumerate(X_ind):
            X.append(np.concatenate((self.U[ind[0]], self.I[ind[1]]))) 

            if i in arm:
                user = ind[0]
                item = ind[1]
        rwd = np.zeros(self.n_arm,dtype=int)
        rwd[arm] = 1

        return np.array(X),X_ind, rwd, arm, user, item    

   