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
from models.rocket_rigid import ridigd_training, scoring, rocketize


def estimate_complexity(input_shape, num_kernels):
    input_length, num_channels = input_shape

    # 估算 generate_kernels 的复杂度
    kernels_flops = num_kernels * (10 + 3)  # 简化估计，包括随机数生成和基本运算

    # 估算 apply_kernels 的复杂度
    avg_kernel_length = 9  # (7 + 9 + 11) / 3
    apply_kernels_flops = input_length * num_channels * num_kernels * avg_kernel_length * 2  # 2 for multiply and add

    # Ridge分类器的复杂度（粗略估计）
    ridge_flops = (2 * num_kernels) ** 2  # 假设特征数为 2 * num_kernels

    total_flops = kernels_flops + apply_kernels_flops + ridge_flops

    # 参数数量
    kernel_params = num_kernels * avg_kernel_length  # 卷积核权重
    kernel_params += num_kernels  # 偏置
    ridge_params = 2 * num_kernels  # Ridge分类器参数（假设二分类问题）

    total_params = kernel_params + ridge_params

    return total_flops, total_params


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

    # 估算复杂度和参数量
    flops, params = estimate_complexity((T_Max, X_tr.shape[2]), num_kernels)
    print(f"Estimated FLOPs: {flops}")
    print(f"Number of parameters: {params}")

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

    return acc, test_f1_rocket, cm, time_collect / X_ts.shape[0], tr_time / X_tr.shape[0], flops, params