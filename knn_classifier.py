import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import cs236781.dataloader_utils as dataloader_utils

from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        # TODO:
        #  Convert the input dataloader into x_train, y_train and n_classes.
        #  1. You should join all the samples returned from the dataloader into
        #     the (N,D) matrix x_train and all the labels into the (N,) vector
        #     y_train.
        #  2. Save the number of classes as n_classes.
        # ====== YOUR CODE: ======
        x_train, y_train = dataloader_utils.flatten(dl_train)
        n_classes = len(dl_train.dataset.source_dataset.classes)
        # ========================

        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = n_classes
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)
        for i in range(n_test):
            # TODO:
            #  - Find indices of k-nearest neighbors of test sample i
            #  - Set y_pred[i] to the most common class among them
            #  - Don't use an explicit loop.
            # ====== YOUR CODE: ======
            ind = torch.argsort(dist_matrix[:, i])[:self.k]  # i is the index of the test -> j for l2 distance matrix
            values, counts = torch.unique(self.y_train[ind], return_counts=True)
            y_pred[i] = values[torch.argmax(counts)]
            # ========================

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    # TODO:
    #  Implement L2-distance calculation efficiently as possible.
    #  Notes:
    #  - Use only basic pytorch tensor operations, no external code.
    #  - Solution must be a fully vectorized implementation, i.e. use NO
    #    explicit loops (yes, list comprehensions are also explicit loops).
    #    Hint: Open the expression (a-b)^2. Use broadcasting semantics to
    #    combine the three terms efficiently.
    #  - Don't use torch.cdist

    dists = None
    # ====== YOUR CODE: ======
    # l2-distance - sqrt((x1-x2)^2)
    # squaring the tensors
    x1_squared = torch.sum(x1**2, dim=1).reshape(-1, 1)  # shape (N1, 1)
    x2_squared = torch.sum(x2**2, dim=1).reshape(-1, 1)  # shape (N2, 1)

    # broadcasting
    x1x2 = torch.matmul(x1, x2.T)  # shape (N1, N2)

    # calculating the l2 distances
    dists = torch.sqrt(torch.add(x1_squared, x2_squared.T) - 2 * x1x2)  # shape (N1, N2)
    # ========================

    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.
    accuracy = None
    # ====== YOUR CODE: ======
    diff = y - y_pred == 0
    accuracy = float(torch.true_divide(sum(diff), len(diff)))
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO:
        #  Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use your train/validation splitter from part 1 (note that
        #  then it won't be exactly k-fold CV since it will be a
        #  random split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        # init for each k
        accuracies.append([])
        num_of_samples = int(len(ds_train) / num_folds)

        dl_train = torch.utils.data.DataLoader(ds_train, num_of_samples)
        x_train, y_train = dataloader_utils.flatten(dl_train)

        for j in range(num_folds):

            # cv = split_data.copy()
            # val_idx = np.sort(cv.pop(j))
            # train_idx = np.sort(np.array(cv).flatten())

            # update indices
            all_ind = np.arange(0, int(len(ds_train)))
            val_ind = np.arange(j * num_of_samples, j * num_of_samples + num_of_samples)
            train_ind = np.setdiff1d(all_ind, val_ind)

            # create valid samplers
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_ind)
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_ind)

            # temporary validation and train sets
            tmp_dl_val = DataLoader(ds_train, batch_size=50, shuffle=False, sampler=valid_sampler)
            tmp_dl_train = DataLoader(ds_train, batch_size=50, shuffle=False, sampler=train_sampler)

            # model training
            model.train(tmp_dl_train)

            # model prediction
            x_val, y_val = dataloader_utils.flatten(tmp_dl_val)
            y_pred = model.predict(x_val)

            # update accuracy list
            accuracies[i].append(accuracy(y_val, y_pred))
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
