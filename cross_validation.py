import numpy as np
from proj1_helpers import *


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def k_fold_cross_validation(y, tx, k_fold, org_l):
    """perform a k-fold cross validation and find the optimal hyper-parameters"""
    seed = 1
    degrees = np.arange(6,7) 
    lambdas = np.logspace(-15, -5, 30) #lambda changing from 10^(-6) to 1
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = np.zeros((len(degrees), len(lambdas)))
    rmse_te = np.zeros((len(degrees), len(lambdas)))
  
    for i_d,d in enumerate(degrees):
        print(i_d)
        for i_l,l in enumerate(lambdas):
            tr_loss = []
            te_loss = []
            for k in range(k_fold):
                
                l_tr, l_te = ridge_reg_cross_validation(y, tx, k_indices, k, l, d, org_l)
                tr_loss.append(l_tr)
                te_loss.append(l_te)
            
            rmse_tr[i_d][i_l] = np.mean(tr_loss)
            rmse_te[i_d][i_l] = np.mean(te_loss)
        
    opt_i = np.argmin(rmse_te)
    opt_d = opt_i // len(lambdas)
    opt_l = opt_i % len(lambdas)
    lambda_ = lambdas[opt_l]
    degree = degrees[opt_d]
    
    
    
    return lambda_, degree

def ridge_reg_cross_validation(y, x, k_indices, k, lambda_, degree, org_l):
    """return the loss of ridge regression."""
    l_test = k_indices[k]
    l_train =  k_indices[~(np.arange(k_indices.shape[0])==k)] 
    l_train = l_train.reshape(-1)
    x_test = x[l_test]
    y_test = y[l_test]
    x_train = x[l_train] 
    y_train = y[l_train] 
    
    
    
    x_train = build_poly(x_train, degree)
    x_test =  build_poly(x_test, degree)
    w, _ = ridge_regression(y_train, x_train, lambda_)
 
    loss_tr = compute_loss(y_train, x_train, w)
    loss_te = compute_loss(y_test, x_test, w) 
    return loss_tr, loss_te



def reg_log_regression_cross_validation(y, x, k_indices, k, lambda_, initial_w, max_iters, gamma):
    """
    Cross validation k-fold for regularized logistic regression
    """
    # splitting the train data to take only the k'th set as test, rest is for training
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    test_set_idx = k_indices[k]
    train_set_idx = np.delete(k_indices, (k), axis=0).ravel()
    x_train = x[train_set_idx, :]
    x_test = x[test_set_idx, :]
    y_train = y[train_set_idx]
    y_test = y[test_set_idx]

    x_train = preprocessing(x_train)
    x_test = preprocessing(x_test)
    weights, _ = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma)
    # predict
    y_train_pred, _ = predict_labels_log_regression(y_train, weights, x_train)
    y_test_pred, _  = predict_labels_log_regression(y_test, weights, x_test)
    
    # compute accuracy
    acc_train = compute_accuracy(y_train_pred, y_train)
    acc_test = compute_accuracy(y_test_pred, y_test)

    return acc_train, acc_test