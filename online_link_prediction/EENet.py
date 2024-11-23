import numpy as np
from EENetClass import Exploitation, Exploration, Decision_maker
import pdb


class EE_Net:
    def __init__(self, dim, n_arm, pool_step_size, lr_1 = 0.01, lr_2 = 0.01, hidden=100, neural_decision_maker = False, kernel_size = 40):
        #Network 1
        self.f_1 = Exploitation(dim, n_arm, pool_step_size, lr_1, hidden)
        
        # number of dimensions of aggregated for f_2  
        f_2_input_dim = self.f_1.total_param // pool_step_size + 1
        #Network 2
        self.f_2 = Exploration(f_2_input_dim, lr_2, kernel_size= kernel_size)

        
        self.arm_select = 0
        
        self.exploit_scores = []
        self.explore_scores = []
        self.ee_scores = []
        self.grad_list = []
        
        self.contexts = []
        self.rewards = []
        self.decision_maker = neural_decision_maker

        #Graph
        self.h = []
        
    def predict(self, context, t):  # np.array 
        self.exploit_scores, self.grad_list = self.f_1.output_and_gradient(context)
        self.explore_scores = self.f_2.output(self.grad_list)
        self.ee_scores = np.concatenate((self.exploit_scores, self.explore_scores), axis=1)

        
        #Neural Greedy and PRB-Greedy:
        # self.arm_select = np.argmax(self.exploit_scores)
        #EEnet and PRB:
        f_2_weight = 1.0
        if t > 500: f_2_weight = 1/np.sqrt(t)
        suml = self.exploit_scores + f_2_weight * (self.explore_scores-1.0)
        self.arm_select = np.argmax(suml)

        #update h : two options 
        #PRB-Greedy
        # self.h = self.exploit_scores
        #PRB
        self.h = suml
        
        return self.arm_select, self.h
    
    def update(self, context, r_1, t):
        # update exploitation network
        self.f_1.update(context[self.arm_select], r_1)
        
        self.contexts.append(context[self.arm_select])
        self.rewards.append(r_1)
        
        # update exploration network
        f_1_predict = self.exploit_scores[self.arm_select][0]
        r_2 = (r_1 - f_1_predict) + 1.0
        self.f_2.update(self.grad_list[self.arm_select], r_2)
        
        # add additional samples to exploration net when the selected arm is not optimal
        if t < 1000:
            if r_1 == 0:
                index = 0
                for i in self.grad_list:
                    c = 1.2
                    if index != self.arm_select:
                        self.f_2.update(i, c)
                    index += 1
    

    def train(self, t):
        #train networks
        loss_1 = self.f_1.train()        
        loss_2 = self.f_2.train()
        return loss_1, loss_2
    