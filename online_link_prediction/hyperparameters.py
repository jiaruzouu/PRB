ALPHA = 0.85
EPSILON = 1e-3

# which method to use
TRACKING_METHOD = 2    # 2 for EvePPR-APP, 3 for EvePPR

# load the transition matrix and prior knowledge or compute from draft. We provide the computed matrices and suggest to load to save time.
LOAD_INSTEAD_OF_RECALCULATION = 1

LOAD_MAX = 33      # 187 timestamps for movielens-1m, 39 timestamps for bitcoinalpha, 33 timestamps for wikilens, 
                    # 20 timestamps for node attribute change
                    # for edge attribute change, 7 for movielens-1m, 11 for bitcoinalpha, 8 for wikilens


# 'movielens-1m', 'bitcoinalpha', 'wikilens'
DATASET_NAME = 'wikilens'


# for ablation study
ABLATION = 0