import os
import torch
from encoders import PointNet_CLS_Encoder, DGCNN_CLS_Encoder
from cls_model_4_compare import get_encoder
import argparse
import numpy as np
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from cls_eval_data import ModelNet40SVM, ScanObjectNNSVM, ModelNet10SVM
from psg_model_3 import Encoder_Head


device = torch.device('cuda')


def get_local_encoder(args):
    if args.model_choice == 0:
        encoder = PointNet_CLS_Encoder().to(device)
    elif args.model_choice == 1:
        encoder = DGCNN_CLS_Encoder().to(device)
    elif args.model_choice == 2:
        encoder = PCT_Encoder().to(device)
    else:
        encoder = Encoder_Head().to(device)
    local_encoder = get_encoder(encoder, args.weight_path)
    return local_encoder.to(device)


def get_loader(args):
    if args.dataset == 'modelnet40':
        train_loader = DataLoader(ModelNet40SVM(partition='train', num_points=args.num_points),
                                  batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(ModelNet40SVM(partition='test', num_points=args.num_points),
                                 batch_size=args.batch_size, shuffle=True)
    elif args.dataset == 'scan':
        train_loader = DataLoader(ScanObjectNNSVM(partition='train', num_points=args.num_points),
                                  batch_size=int(args.batch_size / 2), shuffle=True)
        test_loader = DataLoader(ScanObjectNNSVM(partition='test', num_points=args.num_points),
                                 batch_size=int(args.batch_size / 2), shuffle=True)
    else:
        train_loader = DataLoader(ModelNet10SVM(partition='train', num_points=args.num_points),
                                  batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(ModelNet10SVM(partition='test', num_points=args.num_points),
                                 batch_size=args.batch_size, shuffle=True)
    return train_loader, test_loader


def feature_norm(x):
    # centroid = np.mean(x, axis=1).reshape(x.shape[0], 1, -1)
    centroid = np.mean(x, axis=1).reshape(x.shape[0], -1)
    x = x - centroid
    m = np.max(np.sqrt(np.sum(x**2, axis=1)))
    x = x / m
    return x


def get_feats_labels(loader, feats_t, labels_t, model, args):
    dataset = args.dataset
    for i, (data, label) in enumerate(loader):
        if dataset == "modelnet40":
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
        else:
            labels = label.numpy().tolist()
        data = data.to(device)
        with torch.no_grad():
            feats = model(data)
            # todo: for partseg encoder here
            # feats = torch.max(feats, 2)[0]  # [B, 1024]

        feats = feats.detach().cpu().numpy()
        # feats = feature_norm(feats.squeeze())
        for feat in feats:
            feats_t.append(feat)
        labels_t += labels
    return np.array(feats_t), np.array(labels_t)


def load_feats(args):
    feats_train, labels_train, feats_test, labels_test = [], [], [], []
    model = get_local_encoder(args)
    model = model.eval()

    train_loader, test_loader = get_loader(args)
    feats_train, labels_train = get_feats_labels(train_loader, feats_train, labels_train, model, args)
    feats_test, labels_test = get_feats_labels(test_loader, feats_test, labels_test, model, args)
    feats_train, feats_test = feats_train.squeeze(), feats_test.squeeze()
    print('- Train Feature Shape: ', feats_train.shape)
    print('- Test Feature Shape: ', feats_test.shape)
    return feats_train, labels_train, feats_test, labels_test


def train_svm(args):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    if args.scaler_choice == 0:
        scaler = MinMaxScaler()
    elif args.scaler_choice == 1:
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    scaler_list = ['MinMaxScaler', 'StandardScaler', 'RobustScaler']
    print('- Scaler: ', scaler_list[args.scaler_choice])
    print('- Dataset: ', args.dataset)

    feats_train, labels_train, feats_test, labels_test = load_feats(args)
    scaled = scaler.fit_transform(feats_train)
    model_tl = SVC(C=args.c, kernel='linear')
    print('Start training SVM...')
    # model_tl.fit(feats_train, labels_train)
    model_tl.fit(scaled, labels_train)
    test_scaled = scaler.transform(feats_test)
    print(f"C = {args.c} : {model_tl.score(test_scaled, labels_test)}")
    print('***********************')


if __name__ == "__main__":
    device = torch.device('cuda')
    model_choice = 0  # default
    dataset_choice = ['modelnet40', 'scan']
    batch_size = 128  # default for PointNet
    scaler_choice_list = [0, 1, 2]

    weight_file = r'/home/haruki/下载/SimAttention/scripts/weights/'
    name_list = ['99']
    for name in name_list:
        weight_name = 'cls_dg_0205_cross_multi_' + str(name)
        if weight_name.startswith('cls_dg'):
            model_choice = 1
            batch_size = 32
        weight_path = os.path.join(weight_file, weight_name + '.pth')
        print('For this model[{}]'.format(weight_name))

        for scaler_choice in scaler_choice_list:
            def parameters_init():
                parser = argparse.ArgumentParser(description='Point Cloud Classification Evaluation!')
                parser.add_argument('--model_choice', type=int, default=model_choice, help='0 for PN, 1 for DGCNN')
                parser.add_argument('--weight_path', type=str, default=weight_path)
                parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40', 'scan', 'modelnet10'])
                parser.add_argument('--batch_size', type=int, default=batch_size, help='ScanObjectNN use half')
                parser.add_argument('--num_points', type=int, default=1024)
                parser.add_argument('--scaler_choice', type=int, default=scaler_choice, choices=[0, 1, 2], help='MinMaxScaler, StandardScaler, RobustScaler')
                parser.add_argument('--c', type=float, default=6e-3, help='Linear SVM parameter C, can be tuned')
                parsers = parser.parse_args()
                return parsers

            hyper_parameters = parameters_init()
            train_svm(args=hyper_parameters)
