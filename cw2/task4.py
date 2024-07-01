import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from parameters import (NN_LEARNING_RATE, NN_BATCH_SIZE, NN_EPOCHS,
                        GLOVE_EMBEDDING_DIM)
from utils import (Standardizer, evaluate_test_data, evaluate_validation_data,
                   get_processed_data)


class TwoTowerNet(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.q_layers = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.p_layers = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.final_layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        q_out = self.q_layers(x[:, :self.embedding_dim])
        p_out = self.p_layers(x[:, self.embedding_dim:])
        dot_prod = torch.sum(q_out * p_out, dim=1).reshape(-1, 1)
        q_out_norm = torch.norm(q_out, dim=1).reshape(-1, 1)
        p_out_norm = torch.norm(p_out, dim=1).reshape(-1, 1)
        output = self.final_layers(torch.hstack((dot_prod, q_out_norm, p_out_norm)))
        return output


def neural_net_feature(q_embs, p_embs, standardizer, train=False):
    """Add cosine similarity as feature. Standardize the feature as
    model input."""
    feat = np.hstack([q_embs, p_embs])
    if train:
        stand_feat = standardizer.fit_transform(feat)
    else:
        stand_feat = standardizer.transform(feat)
    return torch.tensor(stand_feat)


def weighted_bce(predictions, targets, pos_weight):
    """Weighted binary cross entropy loss"""
    weights = torch.where(targets == 1, pos_weight, 1)
    losses = F.binary_cross_entropy(predictions, targets)
    loss = torch.sum(weights * losses) / torch.sum(weights)
    return loss


def nn_train(x, y, epochs, lr, batch_size):
    # Training data
    torch.manual_seed(0)
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = TwoTowerNet(GLOVE_EMBEDDING_DIM)
    pos_weight = torch.sum(y == 0) / torch.sum(y == 1)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    loss_hist = []
    for n in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0

        # Mini-batch gradient descent
        for x_batch, y_batch in train_loader:
            optimiser.zero_grad()
            scores = model(x_batch)
            loss = weighted_bce(scores.to('cpu').squeeze(), y_batch, pos_weight)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        elapsed = time.time() - epoch_start
        loss_hist.append(epoch_loss / len(train_loader))
        print(f'Epoch: {n+1:>2}/{epochs}  Loss: {loss_hist[-1]:.6f}  Elapsed: {elapsed:.2f}s')

    return model


def task4():
    print('-'*50 + '\nTask 4\n' + '-'*50)

    # Load embeddings and labels
    print('## Loading training data embeddings ...')
    tr_ids, tr_q_embs, tr_p_embs, tr_y = get_processed_data('train')  # Training data
    print('## Loading validation data embeddings ...')
    va_ids, va_q_embs, va_p_embs, va_y = get_processed_data('valid')  # Validation data

    # Process model input features
    print('## Processing model inputs features')
    tr_y = torch.FloatTensor(tr_y)  # Convert to tensor
    standardizer = Standardizer()
    tr_x = neural_net_feature(tr_q_embs, tr_p_embs, standardizer, train=True)
    va_x = neural_net_feature(va_q_embs, va_p_embs, standardizer)

    # Model training
    print('## Training neural network ...')
    model = nn_train(tr_x, tr_y, NN_EPOCHS, NN_LEARNING_RATE, NN_BATCH_SIZE)

    # Evaluate validation data
    print('## Evaluating the model on validation set ...')
    model.eval()
    nn_pred = lambda x: model(x).detach().flatten()
    spec = {'batch_size': NN_BATCH_SIZE, 'lr': NN_LEARNING_RATE, 'epochs': NN_EPOCHS}
    evaluate_validation_data(va_ids[:, 0], va_x, va_y, nn_pred, 'NN', spec)

    # Evaluate test data
    print('## Evaluating the model on testing set ...')
    get_test_feat = lambda x1, x2: neural_net_feature(x1, x2, standardizer)
    evaluate_test_data(nn_pred, model_name='NN', feat_func=get_test_feat)

    print('## Task 4 completed!')


if __name__ == '__main__':
    task4()
