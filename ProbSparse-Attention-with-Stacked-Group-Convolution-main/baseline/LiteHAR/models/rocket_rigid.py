import os, pickle, time
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import RidgeClassifierCV
import matplotlib.pyplot as plt
from rocket_functions import generate_kernels, apply_kernels
from joblib import Parallel, delayed
from sklearn.metrics import f1_score

def ridigd_training(X, Y):
    model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    model.fit(X, Y)
    return model

def scoring(model, X):
    prediction = model.predict(X)
    return prediction

def rocketize(T_Max,num_kernels,X_tr,X_ts,frequency,Y_tr,Y_ts,reinitialize_rocket):
    input_length = T_Max
    kernels = generate_kernels(input_length, num_kernels)

    print('Rocketizing trianing data ...')
    X_tr_rock = np.zeros((X_tr.shape[0], X_tr.shape[2], 2 * num_kernels))
    for sample_indx in tqdm(range(X_tr.shape[0])):  # for each sample
        input_sample = np.swapaxes(X_tr[sample_indx, :, :], 0, 1)
        X_tr_rock[sample_indx, :, :] = apply_kernels(input_sample, kernels)  # out: (N, 180, 2*N_Kernels)

    print('Rocketizing testing data ...')
    X_ts_rock = np.zeros((X_ts.shape[0], X_ts.shape[2], 2 * num_kernels))
    for sample_indx in tqdm(range(X_ts.shape[0])):  # for each sample
        input_sample = np.swapaxes(X_ts[sample_indx, :, :], 0, 1)
        X_ts_rock[sample_indx, :, :] = apply_kernels(input_sample, kernels)  # out: (N, 180, 2*N_Kernels)

    X_tr = np.swapaxes(X_tr_rock, 1, 2)
    X_ts = np.swapaxes(X_ts_rock, 1, 2)
    return X_tr, X_ts, Y_tr, Y_ts