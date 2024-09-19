import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
import copy
import seaborn as sns
import os
from fvcore.nn import FlopCountAnalysis, parameter_count
from TCN import TemporalConvNet
from math import sqrt
from matplotlib import pyplot as plt


def train_and_test(X_train, X_val, X_test, Y_train, Y_val, Y_test, num_classes, batch_size, n_epochs, name_classes, patience, fold_num, model_state_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将数据转换为PyTorch张量
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    Y_train_tensor = torch.from_numpy(Y_train).long().to(device)
    Y_val_tensor = torch.from_numpy(Y_val).long().to(device)
    Y_test_tensor = torch.from_numpy(Y_test).long().to(device)

    # 定义模型参数
    d_model = 342

    # 实例化模型
    model = Model(num_classes, d_model).to(device)
    # 新增代码：检查是否提供了预训练的模型状态路径
    if model_state_path is not None and os.path.isfile(model_state_path):
        # 加载预训练的模型状态字典
        state_dict = torch.load(model_state_path, map_location=device)

        # 移除分类器全连接层fc1的权重和偏置
        if 'module.fc1.weight' in state_dict:
            del state_dict['module.fc1.weight']
        if 'module.fc1.bias' in state_dict:
            del state_dict['module.fc1.bias']

        # 检查是否是DataParallel模型保存的状态字典
        # 如果是，去除键中的'module.'前缀
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        else:
            new_state_dict = state_dict

        # 加载处理后的状态字典到模型中
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained model parameters from: {model_state_path}, excluding 'fc' layer")
    else:
        # 计算Params：参数数量，即模型中可训练的参数总数
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total Params: {total_params}')

        # 创建一个假的输入张量，模拟实际的输入数据（batch size，seq_length，feature_dim)，以走通一个反向传播从而计算出浮点运算次数
        input_tensor = torch.randn(1, 2000, 342).to(device)

        # 计算Flops：浮点运算次数
        flops = FlopCountAnalysis(model, input_tensor)
        print(f'Total FLOPs: {flops.total()}')

        # 使用DataParallel进行多GPU并行计算
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            model = nn.DataParallel(model)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()

        weight_decay = 5e-4  # 增加权重衰减
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay)

        # 换成学习率衰减
        # factor：减少学习率的因子，即新的学习率 = 原学习率 * factor。
        # patience：在降低学习率前，允许指标多少个epochs没有性能改进，什么性能根据下面看。
        # threshold：测量新的最优值和旧的最优值的显著性差异。
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4,
                                                               threshold=0.0005, cooldown=0, verbose=True)

        # 初始化最佳验证损失和最佳模型状态，用于早停时应用最佳模型状态来测试
        best_val_loss = float('inf')
        best_model = None

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # 训练模型
        for epoch in range(n_epochs):
            model.train()
            train_loss = 0.0
            train_acc = 0.0
            for i in range(0, len(X_train_tensor), batch_size):
                batch_x = X_train_tensor[i:i + batch_size]  # （batch size，seq_length，feature_dim)
                batch_y = Y_train_tensor[i:i + batch_size]
                optimizer.zero_grad()  # 清除（重置）所有被优化参数（模型参数）的梯度。
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()  # 反向传播，计算损失函数对模型参数的梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸，确保梯度的范数不超过一个给定的最大值
                optimizer.step()  # 更新网络参数
                train_loss += loss.item() * batch_x.size(
                    0)  # 累积整个训练集的总损失，loss.item()当前批次的平均损失x批次样本数（batch_x.size(0)）=这个批次的总损失
                predicted = torch.argmax(outputs, dim=1)  # 为每个样本找到预测概率最高的类别
                train_acc += torch.sum((predicted == batch_y).float())  # 计算并累积整个训练集的准确率

            train_loss /= len(X_train_tensor)
            train_acc /= len(X_train_tensor)

            # 在验证集上评估模型
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for i in range(0, len(X_val_tensor), batch_size):
                    batch_x = X_val_tensor[i:i + batch_size]
                    batch_y = Y_val_tensor[i:i + batch_size]
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
                    predicted = torch.argmax(outputs, dim=1)
                    val_acc += torch.sum((predicted == batch_y).float())
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(batch_y.cpu().numpy())

            val_loss /= len(X_val_tensor)
            val_acc /= len(X_val_tensor)
            val_f1 = f1_score(val_labels, val_preds, average='macro')
            # scheduler.step()  # 在验证之后更新学习率

            # 传入监控指标val_loss的值,学习率调度器根据该指标的变化情况来决定是否调整学习率。
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"Epoch [{epoch + 1}/{n_epochs}], "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, "
                  f"Val F1: {val_f1:.4f}")

            # 检查验证损失是否有改善并保存最佳模型状态
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(
                    model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
                patience_counter = 0  # 重置计数器
            else:
                patience_counter += 1  # 增加没有改善的epoch数

            # 检查是否达到了早停的条件
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # 训练结束后，检查是否有保存的最佳模型状态
        if best_model is not None:
            # 如果使用的是 nn.DataParallel，则需要特别处理以加载模型状态
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(best_model)
            else:
                model.load_state_dict(best_model)

        # 绘制训练和验证曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        train_losses_tensor = torch.tensor(train_losses)
        plt.plot(train_losses_tensor.cpu().numpy(), label='Training Loss')
        val_losses_tensor = torch.tensor(val_losses)
        plt.plot(val_losses_tensor.cpu().numpy(), label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()  # 添加一个图例，显示每条线对应的标签

        plt.subplot(1, 2, 2)
        train_accs_tensor = torch.tensor(train_accs)
        plt.plot(train_accs_tensor.cpu().numpy(), label='Training Accuracy')
        val_accs_tensor = torch.tensor(val_accs)
        plt.plot(val_accs_tensor.cpu().numpy(), label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()

        training_curves_path = f'/kaggle/working/training_curves_fold_{fold_num}.png'
        plt.savefig(training_curves_path)
        plt.close()

# 在测试集上评估模型
    model.eval()
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            batch_x = X_test_tensor[i:i + batch_size]
            batch_y = Y_test_tensor[i:i + batch_size]
            outputs = model(batch_x)
            predicted = torch.argmax(outputs, dim=1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())

    test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
    test_f1 = f1_score(test_labels, test_preds, average='macro')

    # 计算混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)

    # 将混淆矩阵规范化，转换为概率形式
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 使用Seaborn绘制混淆矩阵的热力图
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt=".4f", cmap="Blues", xticklabels=name_classes, yticklabels=name_classes)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # 添加以下代码
    confusion_matrix_path = f'/kaggle/working/confusion_matrix_fold_{fold_num}.png'
    plt.savefig(confusion_matrix_path)
    plt.close()
    #Image('confusion_matrix.png')
    print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    return model, test_acc  # 返回模型、测试集准确率和图表路径


class ConvolutionBlock1(nn.Module):
    def __init__(self, d_model, hidden_dim, kernel_size, dropout_rate, groups):
        super(ConvolutionBlock1, self).__init__()
        # 确保d_model和hidden_dim能够被组数整除
        assert d_model % groups == 0, "d_model must be divisible by groups"
        assert hidden_dim % groups == 0, "hidden_dim must be divisible by groups"

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=hidden_dim, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=groups)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=d_model, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=groups)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=groups)

    def forward(self, x):
        x = x.transpose(1, 2)  # 转换为 (batch_size, channels, seq_length) 以适应一维卷积
        x = self.activation(self.conv1(x))
        x = self.channel_shuffle(x, groups=2)  # 确保这里的groups值与init中的一致
        x = self.activation(self.conv2(x))
        x = self.channel_shuffle(x, groups=2)
        x = self.dropout(self.conv3(x))
        x = x.transpose(1, 2)  # 转换回 (batch_size, seq_length, channels)
        return x

    def channel_shuffle(self, x, groups):
        batch_size, num_channels, seq_length = x.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batch_size, groups, channels_per_group, seq_length)
        # transpose
        x = x.transpose(1, 2).contiguous()
        # flatten
        x = x.view(batch_size, -1, seq_length)
        return x

class ConvolutionBlock2(nn.Module):
    def __init__(self, d_model, hidden_dim, kernel_size, dropout_rate, groups):
        super(ConvolutionBlock2, self).__init__()
        # 确保d_model和hidden_dim能够被组数整除
        assert d_model % groups == 0, "d_model must be divisible by groups"
        assert hidden_dim % groups == 0, "hidden_dim must be divisible by groups"

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=hidden_dim, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=groups)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=d_model, kernel_size=kernel_size,
                               padding=kernel_size // 2, groups=groups)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.activation(self.conv1(x)))
        x = self.dropout(self.activation(self.conv2(x)))
        return x


class Model(nn.Module):
    def __init__(self, num_classes, d_model, drop_p=0.3):
        super().__init__()
        self.adaptive_pool1 = nn.AdaptiveAvgPool1d(250)
        self.LN1 = nn.LayerNorm(d_model)  # 加快模型收敛

        self.TCNtransform = TemporalConvNet(num_inputs=342, num_channels=[342, 228, 114], kernel_size=3)
        self.LN2 = nn.LayerNorm(114)
        self.dropout = nn.Dropout(drop_p)  # 添加Dropout层

        #self.GC1 = ConvolutionBlock1(114, 114, 3, 0.3, 2)
        #self.GC2 = ConvolutionBlock2(250, 250, 3, 0.3, 2)

        self.adaptive_pool2 = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(114, num_classes)

    def forward(self, x):
        device = x.device  # 直接从输入张量获取设备信息

        x = x.transpose(1, 2)
        x = self.adaptive_pool1(x)
        x = x.transpose(1, 2)
        x = self.LN1(x)

        x = x.transpose(1, 2)
        x_TCN = self.TCNtransform(x).to(device)
        x_TCN = x_TCN.transpose(1, 2)
        x = x_TCN
        #x = self.GC1(x_TCN)
        #x = self.GC2(x)

        x = self.adaptive_pool2(x.transpose(1, 2)).squeeze(2)  # 使用自适应池化
        x = self.dropout(self.LN2(x))
        x = self.fc1(x)
        return x