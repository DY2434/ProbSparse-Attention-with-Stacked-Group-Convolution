from dataset import *
from UT_HAR_model import *
from NTU_Fi_model import *
from widar_model import *
from self_supervised_model import *
import torch


def load_data_n_model(dataset_name, model_name, root):
    classes = {'UT_HAR_data': 7, 'NTU-Fi-HumanID': 14, 'NTU-Fi_HAR': 6, 'Widar': 22}
    if dataset_name == 'UT_HAR_data':
        print('using dataset: UT-HAR DATA')
        data = UT_HAR_dataset(root)
        train_set = torch.utils.data.TensorDataset(data['X_train'], data['y_train'])
        test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'], data['X_test']), 0),
                                                  torch.cat((data['y_val'], data['y_test']), 0))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True,
                                                   drop_last=True)  # drop_last=True
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
        elif model_name == 'ResNet18':
            print("using model: ResNet18")
            model = UT_HAR_ResNet18()
            train_epoch = 200  # 70
        elif model_name == 'ResNet50':
            print("using model: ResNet50")
            model = UT_HAR_ResNet50()
            train_epoch = 200  # 100
        elif model_name == 'GRU':
            print("using model: GRU")
            model = UT_HAR_GRU()
            train_epoch = 200
        elif model_name == 'BiLSTM':
            print("using model: BiLSTM")
            model = UT_HAR_BiLSTM()
            train_epoch = 200
        elif model_name == 'ViT':
            print("using model: ViT")
            model = UT_HAR_ViT()
            train_epoch = 200  # 100
        return train_loader, test_loader, model, train_epoch


    elif dataset_name == 'NTU-Fi-HumanID':
        print('using dataset: NTU-Fi-HumanID')
        num_classes = classes['NTU-Fi-HumanID']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/test_amp/'),
                                                   batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/train_amp/'),
                                                  batch_size=64, shuffle=False)
        elif model_name == 'ResNet18':
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
        elif model_name == 'ResNet18':
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