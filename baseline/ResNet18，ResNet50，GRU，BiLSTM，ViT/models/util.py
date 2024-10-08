from data.dataset import *
from models.NTU_Fi_model import *
import torch


def load_data_n_model(dataset_name, model_name, root):
    classes = {'NTU-Fi-HumanID': 14, 'NTU-Fi_HAR': 6}

    if dataset_name == 'NTU-Fi-HumanID':
        print('using dataset: NTU-Fi-HumanID')
        num_classes = classes['NTU-Fi-HumanID']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/test_amp/'), #因为下的数据集反了，所以这里用test_amp
                                                   batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/train_amp/'),
                                                  batch_size=64, shuffle=False)

        if model_name == 'ResNet18':
            print("using model: ResNet18")
            model = NTU_Fi_ResNet18(num_classes)
            train_epoch = 50  # 30
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = NTU_Fi_ResNet50(num_classes)
            train_epoch = 50  # 40
        elif model_name == 'GRU':
            print("using model: GRU")
            model = NTU_Fi_GRU(num_classes)
            train_epoch = 50  # 40
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = NTU_Fi_BiLSTM(num_classes)
            train_epoch = 50
        elif model_name == 'ViT':
            print("using model: ViT")
            model = NTU_Fi_ViT(num_classes=num_classes)
            train_epoch = 50
        return train_loader, test_loader, model, train_epoch

    elif dataset_name == 'NTU-Fi_HAR':
        print('using dataset: NTU-Fi_HAR')
        num_classes = classes['NTU-Fi_HAR']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/train_amp/'), batch_size=64,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/test_amp/'), batch_size=64,
                                                  shuffle=False)

        if model_name == 'ResNet18':
            print("using model: ResNet18")
            model = NTU_Fi_ResNet18(num_classes)
            train_epoch = 30
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = NTU_Fi_ResNet50(num_classes)
            train_epoch = 30  # 40
        elif model_name == 'GRU':
            print("using model: GRU")
            model = NTU_Fi_GRU(num_classes)
            train_epoch = 30  # 20
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = NTU_Fi_BiLSTM(num_classes)
            train_epoch = 30  # 20
        elif model_name == 'ViT':
            print("using model: ViT")
            model = NTU_Fi_ViT(num_classes=num_classes)
            train_epoch = 30
        return train_loader, test_loader, model, train_epoch