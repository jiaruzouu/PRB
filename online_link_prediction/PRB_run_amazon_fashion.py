from baselines.load_data import load_amazon_fashion
from EENet import EE_Net
import numpy as np
import os
import scipy.sparse as sp
import utils
from hyperparameters import ALPHA, EPSILON, TRACKING_METHOD, LOAD_INSTEAD_OF_RECALCULATION, LOAD_MAX, DATASET_NAME, ABLATION
from dataprocessing_temporal_movielens import is_in_subgraph, match_from_sub_to_whole, match_from_whole_to_sub 
import pdb
import time
from tqdm import tqdm

def construct_adjacency_matrix_from_G(G, num_users, num_items): 
    num_nodes = num_users + num_items
    A = sp.lil_matrix((num_nodes, num_nodes))
    for i in range(len(G)):
        user, item, weight = G[i]
        if weight == 1:
            A[item + num_users, user] = 1
        else: A[item + num_users , user] = 0
    return A


if __name__ == '__main__':
    dataset = 'amazon_fashion'
    runing_times = 10
    regrets_all = []
    b = load_amazon_fashion()
    
    for i in range(runing_times):  
        lr_1 = 0.001 # learning rate for exploitation network #0.001
        lr_2 = 0.0001 # learning rate for exploration network #0.0005
        alpha = 0.85 
        print(f'-----------------------------------------alpha is {alpha}-------------------------------------')
        epsilon = 1e-6
        regrets = []
        sum_regret = 0
        ee_net = EE_Net(b.dim, b.n_arm, pool_step_size = 50, lr_1 = lr_1, lr_2 = lr_2, hidden=100, neural_decision_maker=False, kernel_size=40)
        G = np.load("./data/Amazon_fashion/new/Insert/amazon_fashion_4000users_noedge.npy")
        num_users = 4000
        num_items = 4000
        v = np.zeros(((num_users + num_items),1))
        A = construct_adjacency_matrix_from_G(G, num_users, num_items)
        A_start = construct_adjacency_matrix_from_G(G, num_users, num_items)
        P = utils.to_prmatrix(A)
        P_new = P.copy()
        h_old = np.zeros((num_users + num_items,1))
        P_dict = {}
        q = {}
        M = {}
        frequency = []
        v_eveppr_list = []
        P_dict[-1] = P
        onehot_ppr = utils.calc_onehot_ppr_matrix(P, alpha, 2).tolil().transpose()
        for i in range(num_users + num_items):
            q[i] = onehot_ppr[i, :]
            M[i] = P_dict[-1]
        v = utils.calc_ppr_by_power_iteration(P, alpha, h_old, 30).ravel()

        for t in tqdm(range(5000)):
            # zero h for each time round
            h = np.zeros((num_users + num_items,1))
            v_observe = np.zeros((10,1))
            h_observe = np.zeros((10,1))
            #do-while structure: ensure the selected item and user is not belonging to those in the 10% given edges
            while True:
                context,context_ind, rwd, arm, user_id, item_id = b.step() 
                if A_start[item_id + num_users, user_id] != 1: break
            # context,context_ind, rwd, arm, user_id, item_id = b.step() 
            arm_select,h_observe = ee_net.predict(context, t) #observe n arms
            #####################################################################################################
            idx_observe = [num_users + context_ind[i][1] for i in range(len(h_observe))] #item id dynamically changing
            # if t < 100: # setting for static h
            h[idx_observe] = h_observe #commend for not updating h dynamically
     
            # EvePPR-complete
            P_dict[t] = P_new.copy()
            v_mid = utils.osp(v, P, P_new, alpha, epsilon, 0)
            delta_h = (h - h_old).ravel()
            for i in range(num_users + num_items):
                if delta_h[i] != 0:
                    q_new = utils.osp(q[i].toarray().ravel(), M[i], P_new, alpha, epsilon, 0)
                    q[i] = sp.lil_matrix(q_new)
                    M[i] = P_dict[t]
                    v_mid = v_mid + delta_h[i] * q_new
            v = v_mid.copy()

            h_old = h.copy()
            # for debugging: exact PPR solution
            # v = utils.calc_ppr_by_power_iteration(P_new,alpha,h,2)
            
            v_observe = v[idx_observe]
            #preserve arm_select here
            arm_select = np.argmax(v_observe) # comment when using EEnet without EvePPR
            #####################################################################################################
            reward = rwd[arm_select]
            regret = np.max(rwd) - reward 
            #####################################################################################################
            #check the correctness updating A and recalculate P
            if arm_select == arm:
                A[item_id + num_users, user_id] = 1 

            P = P_new.copy()
            P_new = P_new.tolil()
            P_new[:, user_id] = A[:, user_id]/A[:, user_id].sum() 
            P_new = P_new.tocsr()

            #####################################################################################################
            #update parameters
            ee_net.update(context, reward, t)


            # train the net

            sum_regret += regret
            if t<1000:
                if t%50 == 0:
                    loss_1,loss_2 = ee_net.train(t)

            else:
                if t%100 == 0:
                    loss_1,loss_2 = ee_net.train(t)
                    
            regrets.append(sum_regret)
            if t % 50 == 0:
                # print('round:{}, regret: {:},  average_regret: {:.3f}, loss_1:{:.4f}, loss_2:{:.4f}, loss_3:{:.4f}'.format(t,sum_regret, sum_regret/(t+1), loss_1, loss_2, loss_3)) 
                print('round:{}, regret: {:},  average_regret: {:.3f}, loss_1:{:.4f}, loss_2:{:.4f}'.format(t,sum_regret, sum_regret/(t+1), loss_1, loss_2))   
                # print('round:{}, regret: {:},  average_regret: {:.3f}, loss_1:{:.4f}'.format(t,sum_regret, sum_regret/(t+1), loss_1)) 
        print(' regret: {:},  average_regret: {:.2f}'.format(sum_regret, sum_regret/(t+1)))
        regrets_all.append(regrets)
    path = os.getcwd()     
    np.save('{}/results/PRB/10%_Graph_Input/Amazon_fashion/4000/PRB2_regrets.npy'.format(path), regrets_all)

    

