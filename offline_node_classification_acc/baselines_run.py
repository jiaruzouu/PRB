from KernelUCB import KernelUCB
from LinUCB import Linearucb
from Neural_epsilon import Neural_epsilon
from NeuralTS import NeuralTS
from NeuralUCB import NeuralUCBDiag
from NeuralNoExplore import NeuralNoExplore
import argparse
import numpy as np
import sys 
from tqdm import tqdm, trange 
import time 
from load_data import load_cora,load_citeseer, load_movielen,load_facebook, load_grqc, load_amazon,load_amazon_fashion,load_pubmed, load_cora_train,load_cora_test, load_citeseer_train,load_citeseer_test, load_pubmed_train, load_pubmed_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run baselines')
    parser.add_argument('--dataset', default='yelp', type=str, help='mnist, yelp, movielens, disin')
    parser.add_argument("--method", nargs="+", default=["NeuralTS"], help='list: ["NeuralTS"]')
    parser.add_argument('--lamdba', default='0.1', type=float, help='Regulization Parameter')
    parser.add_argument('--nu', default='0.001', type=float, help='Exploration Parameter')
    
    args = parser.parse_args()
    dataset = args.dataset
    arg_lambda = args.lamdba 
    arg_nu = args.nu
    
    print("running methods:", args.method)
    for method in args.method:
        inference_time = []
        regrets_all = []
        for i in range(1):
            b = load_cora_train()
            print("Dataset: Cora")
            print("Training dataset size is:", len(b.X_all))
            
            if method == "KernelUCB":
                model = KernelUCB(b.dim, arg_lambda, arg_nu)

            elif method == "LinUCB":
                model = Linearucb(b.dim, arg_lambda, arg_nu)

            elif method == "Neural_epsilon":
                epsilon = 0.01
                model = Neural_epsilon(b.dim, epsilon)

            elif method == "NeuralTS":
                print("method is: ", method)
                model = NeuralTS(b.dim, b.n_arm, m = 100, sigma = arg_lambda, nu = arg_nu)

            elif method == "NeuralUCB":
                print("method is: ", method)
                model = NeuralUCBDiag(b.dim, lamdba = arg_lambda, nu = arg_nu,  hidden = 100)
                
            elif method == "NeuralNoExplore":
                model = NeuralNoExplore(b.dim)
            else:
                print("method is not defined. --help")
                sys.exit()
                
            regrets = []
            sum_regret = 0
            print("Round; Regret; Regret/Round")
            rounds = 10000
            for t in range(rounds):
                '''Draw input sample'''
                tic = time.perf_counter()
                context,context_ind, rwd, arm, user_id, item_id = b.step()
                arm_select = model.select(context)
                reward = rwd[arm_select]

                if method == "LinUCB" or method == "KernelUCB":
                    model.train(context[arm_select],reward)

                elif method == "Neural_epsilon" or method == "NeuralUCB" or method == "NeuralTS" or method == "NeuralNoExplore":
                    model.update(context[arm_select], reward)
                    toc = time.perf_counter()
                    inference_time.append(toc-tic)
                    if t<1000:
                        if t%50 == 0:
                            loss = model.train(t)
                    else:
                        if t%100 == 0:
                            loss = model.train(t)

                regret = np.max(rwd) - reward
                sum_regret+=regret
                regrets.append(sum_regret)
                if t % 50 == 0:
                    print('{}: {:}, {:.4f}'.format(t, sum_regret, sum_regret/(t+1)))
          
                # offline Testing 
                if t%1000 == 0 or t == rounds - 1:
                    test_b = load_cora_test()  
                    print("Testing dataset size is:", len(test_b.X_all))
                    correct = 0 
                    for i in trange(len(test_b.X_all)):
                        context,context_ind, rwd, arm, user_id, item_id = test_b.step()
                        arm_select = model.select(context)
                        if arm_select == arm:
                            correct += 1
                    print("The testing accuracy is: ",correct/len(test_b.X_all))
                
            print("run:", i, "; ", "regret:", sum_regret)
            regrets_all.append(regrets)
    
    
    
    
    
    
    
    
    
    
    
    
        