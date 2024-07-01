# ---------------------------------
# Data Processing
# ---------------------------------
# Number of irrelevant records kept for each training query
IRREL_KEEP = 2

# Embeddings. See https://pytorch.org/text/stable/vocab.html#glove
GLOVE_MODEL = '42B'
GLOVE_EMBEDDING_DIM = 300


# ---------------------------------
# Task 2 Logistic Regression
# ---------------------------------
# Evaluate effects of different learning rate
LR_LEARNING_RATES = [1e-6, 1e-5, 1e-4, 1e-3]

# Model training params
LR_BATCH_SIZE = 64
LR_TOLERANCE = 1e-6
LR_EPOCHS = 2000


# ---------------------------------
# Task 3 LambdaMART
# ---------------------------------
# Grid search hyperparameter optimisation
XGB_PARAMS_GRID = {
    'eta': [0.01, 0.05, 0.1, 0.3],   # Learning rate
    'max_depth': [3, 4, 5],          # Tree depth
    'min_child_weight': [1, 2, 3],   # ~Minimum instances for split
    'lambda': [0, 0.05, 0.1],        # L2 regularisation
    'subsample': [0.7, 0.85, 1],     # Row subsampling
}
XGB_N_FOLD = 5

# Model training params
XGB_NUM_BOOST_ROUND = 10


# ---------------------------------
# Task 4 Neural Network
# ---------------------------------
# Model training params
NN_BATCH_SIZE = 1024
NN_LEARNING_RATE = 1e-3
NN_EPOCHS = 30
