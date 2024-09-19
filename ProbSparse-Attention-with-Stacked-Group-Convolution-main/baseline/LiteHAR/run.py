import argparse
import numpy as np
import sys
sys.path.insert(0, "utils")
sys.path.insert(0, "models")
from train_and_test import train_and_test
from utils.dataloader import CSI_Dataset

def main():
    # 设置配置和默认值
    parser = argparse.ArgumentParser()
    parser.add_argument("-cv", "--num_runs", type=int, default=1)
    parser.add_argument("-m", "--model", default="rigRocket")
    parser.add_argument("-k", "--num_kernels", type=int, default=1000)
    parser.add_argument("-e", "--num_epochs", type=int, default=100)
    parser.add_argument("-g", "--gpu", type=int, default=0)

    # 解析命令行参数
    args = parser.parse_args()

    # 加载新数据集
    train_dataset = CSI_Dataset("/kaggle/input/ntu-fi-har/NTU-Fi_HAR/train_amp")
    test_dataset = CSI_Dataset("/kaggle/input/ntu-fi-har/NTU-Fi_HAR/test_amp")
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    # 提取X和Y
    X_train, Y_train = train_dataset.data, train_dataset.labels
    X_test, Y_test = test_dataset.data, test_dataset.labels

    # 更多参数设置
    N_cv = args.num_runs
    N_classes = 7
    pooling = 1
    frequency = '{}hz'.format('1k' if pooling == 1 else int(1000 / pooling))

    # partial_flag=False 操作是否完整，处理一部分还是全部 lr学习率 decayrate学习率衰减的速率
    acc, test_f1_rocket, cm, time_collect, tr_time = train_and_test(X_train, X_test, Y_train, Y_test,
                                                                        args.num_kernels, pooling=pooling,
                                                                        frequency=frequency, reinitialize_rocket=True,
                                                                        model_='rigRocket')

    print("\nModel: {}".format(args.model))
    print("Training Accuracy : {:.2f}%".format(acc * 100))
    print("Test F1 score (Transformer): {:.2f}%".format(test_f1_rocket * 100))
    print("Testing Accuracy : ", time_collect)
    print("Testing Accuracy : ", tr_time)

if __name__ == "__main__":
    main()



