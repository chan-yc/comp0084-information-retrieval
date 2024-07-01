import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from parameters import (LR_BATCH_SIZE, LR_EPOCHS, LR_LEARNING_RATES,
                        LR_TOLERANCE)
from utils import (f1_score, get_processed_data, evaluate_test_data,
                   evaluate_validation_data, ASSET_DIR, Standardizer)


class LogisticRegression:
    """Logistic Regression model for binary classification. The model
    uses weighted cross-entropy loss and gradient descent for training.
    It is designed to work with numpy array.
    """

    def __init__(self, dim, lr):
        # Initilise model weights with zeros
        self.w = np.zeros(dim)
        # Learning rate
        self.lr = lr

    def forward(self, x):
        """Batch forward pass through the model."""
        return 1 / (1 + np.exp(-x @ self.w))

    def step(self, x, y, y_pred, label_w):
        """Update weights using gradient descent."""
        self.w -= self.lr * self.grad(x, y, y_pred, label_w)

    @staticmethod
    def loss(y, y_pred, label_w):
        """Weighted mean of batch Cross Entropy Loss."""
        return - label_w @ (y*np.log(y_pred) + (1-y)*np.log(1-y_pred)) / label_w.sum()

    @staticmethod
    def grad(x, y, y_pred, label_w):
        """Gradient of the loss w.r.t. the weights."""
        return (label_w * (y_pred-y)) @ x / label_w.sum()

    def __call__(self, *args, **kwargs):
        """Alias for forward (Like in PyTorch)"""
        return self.forward(*args, **kwargs)


def log_reg_features(q_embs, p_embs, standardizer, train=False):
    """Standardize the embeddings vectors and add bias term."""
    feat = np.hstack([q_embs, p_embs])
    if train:
        stand_feat = standardizer.fit_transform(feat)
    else:
        stand_feat = standardizer.transform(feat)
    bias = np.ones(q_embs.shape[0]).reshape(-1, 1)
    return np.hstack([bias, stand_feat])


def log_reg_train(x, y, epochs, lr, tol, batch_size):
    """Train a logistic regression model using mini-batch gradient descent."""
    print(f'## Training model [learning rate={lr}, epochs={epochs}, tol={tol}] ...')
    model = LogisticRegression(x.shape[1], lr)
    loss_hist = []

    # Data weights
    pos_weight = np.sum(y == 0) / np.sum(y == 1)
    label_w = np.where(y == 1, pos_weight, 1)

    # Training loop
    early_stop = False
    for n in tqdm(range(epochs)):
        loss = 0

        # Mini-batch gradient descent
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            y_batch = y[i:i+batch_size]
            label_w_batch = label_w[i:i+batch_size]
            y_pred = model(x_batch)
            model.step(x_batch, y_batch, y_pred, label_w_batch)
            loss += model.loss(y_batch, y_pred, label_w_batch)

        # Epoch mean loss
        loss /= (len(x) / batch_size)  # Divide by number of batches
        loss_hist.append(loss)

        # Early stopping if relative improvement is small
        if n > 0 and np.abs(loss_hist[n-1] - loss) / loss_hist[n-1] < tol:
            early_stop = True
            break

    if early_stop:
        print(f'Converged at epoch {n+1}. Early stopping. (loss: {loss_hist[-1]:.4f})')
    else:
        print(f'Completed {epochs} epochs. Not yet converged. (loss: {loss_hist[-1]:.4f})')

    return model, loss_hist


def task2():
    print('-'*50 + '\nTask 2\n' + '-'*50)
    np.random.seed(11)

    # Load embeddings and labels
    print('## Loading training data embeddings ...')
    _, tr_q_embs, tr_p_embs, tr_y = get_processed_data('train')  # Training data
    print('## Loading validation data embeddings ...')
    va_ids, va_q_embs, va_p_embs, va_y = get_processed_data('valid')  # Validation data

    # Process model input features
    print('## Processing model inputs features ...')
    standardizer = Standardizer()
    tr_x = log_reg_features(tr_q_embs, tr_p_embs, standardizer, train=True)
    va_x = log_reg_features(va_q_embs, va_p_embs, standardizer)

    # Shuffle training data
    np.random.seed(1)
    shuffle = np.random.choice(len(tr_x), size=len(tr_x), replace=False)
    tr_x, tr_y = tr_x[shuffle], tr_y[shuffle]

    # Train model with different learning rates
    models = [log_reg_train(tr_x, tr_y, LR_EPOCHS, lr, LR_TOLERANCE, LR_BATCH_SIZE)
              for lr in LR_LEARNING_RATES]

    # Plot learning curves
    plt.rcParams['figure.figsize'] = [6.4, 4.8]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['font.size'] = 12
    print('## Plotting and saving learning curves ...')
    for lr, (_, loss) in zip(LR_LEARNING_RATES, models):
        plt.plot(range(1, len(loss)+1), loss, label=f'lr = {lr:1.0e}')
        plt.annotate(f'{len(loss)}', (len(loss), loss[-1]), xytext=(0, 5),
                     textcoords="offset points", ha='center', c='grey', fontsize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Epoch Loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(ASSET_DIR, 'LR_learning_curves.pdf'))

    # Best model - highest F1 on training set
    val_scores = [(idx, f1_score(tr_y, np.round(model(tr_x))))
                  for idx, (model, _) in enumerate(models)]
    best_idx = max(val_scores, key=lambda x: x[1])[0]
    best_lr = LR_LEARNING_RATES[best_idx]
    best_model = models[best_idx][0]

    # Evaluate validation data
    print('## Evaluating best models on validation set ...')
    specs = {'best_lr': best_lr, 'max_epochs': LR_EPOCHS,
             'tolerance': LR_TOLERANCE, 'batch_size': LR_BATCH_SIZE}
    evaluate_validation_data(va_ids[:, 0], va_x, va_y, best_model, 'LR', specs)

    # Evaluate test data
    print('## Evaluating best models on testing set ...')
    get_test_feat = lambda x1, x2: log_reg_features(x1, x2, standardizer)
    evaluate_test_data(best_model, model_name='LR', feat_func=get_test_feat)

    print('## Task 2 completed!')


if __name__ == '__main__':
    task2()
