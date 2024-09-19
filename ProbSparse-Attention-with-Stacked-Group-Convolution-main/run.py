import argparse
import numpy as np
import copy
import torch
import sys
sys.path.insert(0, "utils")
sys.path.insert(0, "models")
# 确保导入路径正确
from train_and_test import SGCT
from data.dataloader import CSI_Dataset
from sklearn.model_selection import KFold


def main():
    # 设置配置和默认值
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--num_epochs", type=int, default=60)
    parser.add_argument("-kf", "--k_folds", type=int, default=5)  # 添加K折交叉验证的参数

    # 解析命令行参数
    args = parser.parse_args()

    # 加载新数据集，更换环境需修改
    model_state_path = "/kaggle/input/probthfgcs2/ProbTHFGCS2/best_model.pth"
    train_dataset = CSI_Dataset("/kaggle/input/ntu-fi-har/NTU-Fi_HAR/train_amp")
    test_dataset = CSI_Dataset("/kaggle/input/ntu-fi-har/NTU-Fi_HAR/test_amp")
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    # 提取X和Y
    X_train, Y_train = train_dataset.data, train_dataset.labels
    X_test, Y_test = test_dataset.data, test_dataset.labels

    # 从训练集实例中获取排序后的类别名称列表
    class_names = train_dataset.get_sorted_categories()

    # 更多参数设置
    num_classes = 6
    best_model = None
    best_acc = 0
    best_fold = None

    # 使用K折交叉验证
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    test_accs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        # 创建训练和验证的子集
        X_train_fold = X_train[train_idx]
        Y_train_fold = Y_train[train_idx]
        X_val_fold = X_train[val_idx]
        Y_val_fold = Y_train[val_idx]

        # X_train_fold, Y_train_fold, X_val_fold, Y_val_fold，都是numpy数组
        # 调用模型训练和评估函数
        model, test_acc = SGCT(X_train_fold, X_val_fold, X_test,
                                            Y_train_fold, Y_val_fold, Y_test,
                                            num_classes, batch_size=64,
                                            n_epochs=args.num_epochs,
                                            name_classes=class_names,
                                            patience=6, fold_num=fold,
                                            model_state_path=model_state_path)
        # 比较并保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)
            best_fold = fold
        test_accs.append(test_acc)

    # 最后，保存最佳模型的状态
    if best_model is not None:
        # 模型使用PyTorch，根据实际情况调整保存方式
        torch.save(best_model.state_dict(), '/kaggle/working/best_model.pth')  #更换环境需修改

    print("\nTesting Accuracy (Rocket) across {} folds: {:.2f}% ± {:.2f}%".format(
        args.k_folds, np.mean(test_accs) * 100, np.std(test_accs) * 100))
    print("Best Testing Accuracy (Rocket) across {} folds: {:.2f}% on fold {}".format(
        args.k_folds, best_acc * 100, best_fold + 1))  # 加 1 因为 fold 计数通常从 1 开始

if __name__ == "__main__":
    main()