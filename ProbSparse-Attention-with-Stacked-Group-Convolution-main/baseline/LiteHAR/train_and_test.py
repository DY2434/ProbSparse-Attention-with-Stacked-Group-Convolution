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
from models import ridigd_training, scoring, rocketize

def train_and_test(X_tr, X_ts, Y_tr, Y_ts, num_kernels, pooling, frequency, reinitialize_rocket, model_):
    #### Sampling along time
    print('Sampling Frequency is:', frequency)
    if pooling > 1:
        print('Sampling along time at window size of ', str(pooling), ' ...')
        X_tr = X_tr[:, ::pooling, :]
        X_ts = X_ts[:, ::pooling, :]
        T_Max = X_tr.shape[1]
    T_Max = X_tr.shape[1]
    print(T_Max)
    print(X_tr.shape)
    st = time.time()

    X_tr, X_ts, Y_tr, Y_ts = rocketize(T_Max, num_kernels, X_tr, X_ts, frequency, Y_tr, Y_ts, reinitialize_rocket)
    print(X_tr.shape, X_ts.shape)  # N,2xKernel, 90

    print('Parallel Training ...')
    Nsubc = X_tr.shape[2]
    models = Parallel(n_jobs=-2, backend="threading")(
        delayed(ridigd_training)(X_tr[:, :, m_], Y_tr) for m_ in tqdm(range(Nsubc)))
    tr_time = time.time() - st

    # Testing
    print('Parallel Testing ...')
    top_collection = []
    disagrees_subcarries_collect = []
    disagrees_histogram = np.zeros((1, Nsubc))
    time_collect = 0
    for s_indx in range(X_ts.shape[0]):  # for each test sample
        st = time.time()
        predictions = Parallel(n_jobs=1, backend="threading")(
            delayed(scoring)(models[m_], np.expand_dims(X_ts[s_indx, :, m_], axis=0)) for m_ in range(Nsubc))
        time_collect += (time.time() - st)
        (unique, counts) = np.unique(predictions, return_counts=True)
        top_collection.append([unique[np.argmax(counts)], Y_ts[s_indx]])  # prediction Target
        disagrees_binary = predictions != Y_ts[s_indx]
        disagrees_subcarries = np.where(disagrees_binary == True)[0]
        disagrees_subcarries_collect.append(disagrees_subcarries)
        for i in disagrees_subcarries:  # histogram of disagrees update
            disagrees_histogram[0, i] += 1

    print('Prediction vs. Target:', top_collection)
    print('Disagreed subcarriers histogram:', disagrees_histogram / X_ts.shape[0])
    top_collection = np.asarray(top_collection)
    acc = (np.sum(top_collection[:, 0] == top_collection[:, 1])) / X_ts.shape[0]
    print('Accuracy is:', acc)
    test_f1_rocket = f1_score(top_collection[:, 1], top_collection[:, 0], average='macro')
    print('Testing F1 Score of Rocket Model is:', test_f1_rocket)
    print('Avg. Inferene Time (full,per sample):', time_collect, time_collect / X_ts.shape[0])
    print('Training Time (full,per sample):', tr_time, tr_time / X_tr.shape[0])
    cm = confusion_matrix(top_collection[:, 1], top_collection[:, 0])  # Target prediction

    return acc, test_f1_rocket, cm, time_collect / X_ts.shape[0], tr_time / X_tr.shape[0]