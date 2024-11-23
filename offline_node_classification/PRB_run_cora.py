from baselines.load_data import load_mnist_1d, load_yelp, load_movielen, load_cora, load_cora_train,load_cora_test
from EENet import EE_Net
import numpy as np
import os
import scipy.sparse as sp
import utils
from hyperparameters import ALPHA, EPSILON, TRACKING_METHOD, LOAD_INSTEAD_OF_RECALCULATION, LOAD_MAX, DATASET_NAME, ABLATION
from dataprocessing_temporal_movielens import is_in_subgraph, match_from_sub_to_whole, match_from_whole_to_sub #TODO: need to add other datasets later
import pdb
import time
from tqdm import tqdm,trange
import torch

def construct_adjacency_matrix_from_G(G, num_users, num_class): #TODO: initlize to no edge 
    num_nodes = num_users + num_class
    A = sp.lil_matrix((num_nodes, num_nodes))
    return A

def construct_no_edge_graph(G):
    G_origin = torch.transpose(G,0,1).numpy()
    zeros = np.zeros((G_origin.shape[0], 1)) 
    G = np.concatenate((G_origin, zeros), axis=1)
    return G_origin, G

def connect_current_nodes_edges(A, G_dict, current_nodes, new_node) :
    if len(current_nodes) == 0 : return A
    for current_node in current_nodes:
        edge1, edge2 = [current_node, new_node], [new_node, current_node]
        for edge in G_dict[new_node]: 
            edge = edge.tolist()
            # pdb.set_trace()
            if edge1 == edge or edge2 == edge:
                A[current_node, new_node] = 1
                A[new_node, current_node] = 1
    
    return A 

def build_graph_dict(G):
    graph_dict = {}
    
    # Iterate through each edge
    for edge in G:
        node1, node2 = edge
        
        # Add edge to node1's list of edges
        if node1 not in graph_dict:
            graph_dict[node1] = []
        graph_dict[node1].append(edge)
        
        # Add edge to node2's list of edges
        if node2 not in graph_dict:
            graph_dict[node2] = []
        graph_dict[node2].append(edge)
    
    return graph_dict
    
    
if __name__ == '__main__':
    dataset = 'cora'
    runing_times = 10
    regrets_all = []
    rounds = 10000
    
    for i in range(runing_times):  
        lr_1 = 0.01 # learning rate for exploitation network #0.001
        lr_2 = 0.001 # learning rate for exploration network #0.0005
        alpha = 0.85 
        epsilon = 1e-6
        regrets = []
        sum_regret = 0
        b = load_cora_train()
        test_b = load_cora_test()
        print("Training dataset size is:", len(b.X_all))
        print("Testing dataset size is:", len(test_b.X_all))
        print("Training rounds are:", rounds)
        print(f'-----------------------------------------alpha is {alpha}-------------------------------------')
        ee_net = EE_Net(b.dim, b.n_arm, pool_step_size = 50, lr_1 = lr_1, lr_2 = lr_2, hidden=500, neural_decision_maker=False, kernel_size=40)
        G_origin, G = construct_no_edge_graph(b.edge_index_all)
        G_dict = build_graph_dict(G_origin)
        num_users = 2708
        num_class = b.n_arm
        v = np.zeros(((num_users + num_class),1))
        A = construct_adjacency_matrix_from_G(G, num_users, num_class)
        P = utils.to_prmatrix(A)
        P_new = P.copy()
        h_old = np.zeros((num_users + num_class, 1))
        P_dict = {}
        current_nodes = []
        q = {}
        M = {}
        frequency = []
        v_eveppr_list = []
        P_dict[-1] = P
        onehot_ppr = utils.calc_onehot_ppr_matrix(P, alpha, 2).tolil().transpose()
        for i in range(num_users + num_class):
            q[i] = onehot_ppr[i, :]
            M[i] = P_dict[-1]
        v = utils.calc_ppr_by_power_iteration(P, alpha, h_old, 30).ravel()

        for t in range(rounds): 
            h = np.zeros((num_users + num_class, 1))
            v_observe = np.zeros((num_class,1))
            h_observe = np.zeros((num_class,1))
            context,context_ind, rwd, arm, user_id, item_id = b.step() 
            A = connect_current_nodes_edges(A, G_dict, current_nodes, user_id)
            current_nodes.append(user_id)
            
            arm_select,h_observe = ee_net.predict(context, t) #observe n arms
            idx_observe = [context_ind[i][1] for i in range(len(h_observe))]
            h[idx_observe] = h_observe
            P_dict[t] = P_new.copy()
            v_mid = utils.osp(v, P, P_new, alpha, epsilon, 0)
            delta_h = (h - h_old).ravel()
            for i in range(num_users + num_class):
                if delta_h[i] != 0:
                    q_new = utils.osp(q[i].toarray().ravel(), M[i], P_new, alpha, epsilon, 0)
                    q[i] = sp.lil_matrix(q_new)
                    M[i] = P_dict[t]
                    v_mid = v_mid + delta_h[i] * q_new
            v = v_mid.copy()
            
            h_old = h.copy()
            v_observe = v[idx_observe]
            #preserve arm_select here
            arm_select = np.argmax(v_observe) 
            reward = rwd[arm_select]
            regret = np.max(rwd) - reward 
            
            if arm_select == arm:
                A[item_id, user_id] = 1
                A[user_id, item_id] = 1

            P = P_new.copy()
            P_new = P_new.tolil()
            P_new[:, user_id] = A[:, user_id]/A[:, user_id].sum() 
            P_new = P_new.tocsr()
            
            ee_net.update(context, reward, t)
            # train the net
            sum_regret += regret
            if t<1000:
                if t% 50 == 0:
                    loss_1,loss_2 = ee_net.train(t)

            else:
                if t% 100 == 0: 
                    loss_1,loss_2 = ee_net.train(t)
                    
            regrets.append(sum_regret)
            if t % 50 == 0:
                print('round:{}, regret: {:},  average_regret: {:.3f}, loss_1:{:.4f}, loss_2:{:.4f}'.format(t,sum_regret, sum_regret/(t+1), loss_1, loss_2))   
                
            if t%1000 == 0 or t == rounds - 1:  # Save the model after 10000 rounds
                correct = 0 
                correct_pagerank = 0
                test_b = load_cora_test()
                num_users_test = 2708
                num_class_test = test_b.n_arm
                
                G_origin_test, G_test = construct_no_edge_graph(test_b.edge_index_all)
                G_dict_test = build_graph_dict(G_origin_test)
                v_test = np.zeros(((num_users_test + num_class_test),1))
                A_test = construct_adjacency_matrix_from_G(G_test, num_users_test, num_class_test)
                P_test = utils.to_prmatrix(A_test)
                P_new_test = P_test.copy()
                h_old_test = np.zeros((num_users_test + num_class_test, 1))
                P_dict_test = {}
                current_nodes_test = []
                q_test = {}
                M_test = {}
                frequency_test = []
                v_eveppr_list_test = []
                P_dict_test[-1] = P_test
                onehot_ppr_test = utils.calc_onehot_ppr_matrix(P_test, alpha, 2).tolil().transpose()
                for i in range(num_users_test + num_class_test):
                    q_test[i] = onehot_ppr_test[i, :]
                    M_test[i] = P_dict_test[-1]
                v_test = utils.calc_ppr_by_power_iteration(P_test, alpha, h_old_test, 30).ravel()
                
                for i in trange(len(test_b.X_all)):
                    context,context_ind, rwd, arm, user_id, item_id = test_b.step() 
                    h_test = np.zeros((num_users_test + num_class_test, 1))
                    v_observe_test = np.zeros((num_class_test,1))
                    h_observe_test = np.zeros((num_class_test,1))
                    A_test = connect_current_nodes_edges(A_test, G_dict_test, current_nodes_test, user_id)
                    current_nodes_test.append(user_id)
                    arm_select,h_observe_test = ee_net.predict(context, t)
                    if arm_select == arm:
                        correct += 1
                    
                    idx_observe_test = [context_ind[i][1] for i in range(len(h_observe_test))]
                    h_test[idx_observe_test] = h_observe_test
                    P_dict_test[t] = P_new_test.copy()
                    v_mid_test = utils.osp(v_test, P_test, P_new_test, alpha, epsilon, 0)
                    delta_h_test = (h_test - h_old_test).ravel()
                    
                    for i in range(num_users_test + num_class_test):
                        if delta_h_test[i] != 0:
                            q_new_test = utils.osp(q_test[i].toarray().ravel(), M_test[i], P_new_test, alpha, epsilon, 0)
                            q_test[i] = sp.lil_matrix(q_new_test)
                            M_test[i] = P_dict_test[t]
                            v_mid_test = v_mid_test + delta_h_test[i] * q_new_test
                    v_test = v_mid_test.copy()
                    
                    h_old_test = h_test.copy()
                    v_observe_test = v[idx_observe_test]
                    arm_select_new = np.argmax(v_observe_test)
                    # pdb.set_trace()
                    if arm_select_new == arm:
                        correct_pagerank += 1
                    
                    P_test = P_new_test.copy()
                    P_new_test = P_new_test.tolil()
                    P_new_test[:, user_id] = A_test[:, user_id]/A_test[:, user_id].sum() 
                    P_new_test = P_new_test.tocsr()
                    
                        
                print("The testing accuracy is: ", correct/len(test_b.X_all))

        print(' regret: {:},  average_regret: {:.2f}'.format(sum_regret, sum_regret/(t+1)))
        regrets_all.append(regrets) 