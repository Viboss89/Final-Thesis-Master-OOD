import tqdm
import torch
from torch.utils.data import DataLoader
from utils.config import DefaultConfig
from models.net_builder import net_builder
from dataprepare.dataloader import DatasetCFP
from torch.nn import functional as F
from sklearn import metrics
from dataprepare.dataloader import class_sampler, Folders_dataset
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, mahalanobis

def val_cosine(train_dataloader, val_dataloader, model, args, mode, device, Type=True):
    print('\n')
    print('====== Start {} ======!'.format(mode))
    model_features = net_builder(args.net_work, args.num_classes).to(device)
    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    # load trained model for test
    print("=> loading trained model '{}'".format(args.trained_model_path))
    checkpoint = torch.load(
        args.trained_model_path)
    model_features.load_state_dict(checkpoint['state_dict'])
    print('Done!')


    model_features.affine_classifier = torch.nn.Sequential()
    model.eval()
    model_features.eval()


    # TODO calcular threshold cosine i mahalanobis utilitzant train (per a mitjanes) i val per a trobar el threshold
    # utilitzant un mètode similar a Dirichlet
    u_list = []
    u_label_list = []

    features_list = []

    # calculate mean and inverse covariance from train set
    tbar = tqdm.tqdm(train_dataloader, desc='\r') 

    with torch.no_grad():
        for i, img_data_list in enumerate(tbar):
            Fundus_img = img_data_list[0].to(device)
            cls_label = img_data_list[1].long().to(device)
            pred = model_features.forward(Fundus_img)
            for i, label in enumerate(cls_label):
                features_list.append([label.cpu().numpy(), pred[i].cpu().numpy()])

    features_list = np.array(features_list)
    features_list = np.hstack((np.vstack(features_list[:,0]),np.vstack(features_list[:,1])))

    """
    X_embedded = PCA(n_components=2).fit_transform(features_list[:,1:])
    fig, ax = plt.subplots()
    for g in np.unique(features_list[:,0].astype(int)):
        ix = np.where(np.vstack(features_list[:,0]) == g)
        ax.scatter(X_embedded[ix,0], X_embedded[ix,1], label = [1,2,3,4][g])
    ax.legend()
    plt.show()
    """
    mean = []
    cov = []
    for i in range(args.num_classes):
        temp = features_list[features_list[:,0]==i]
        mean.append(np.mean(temp[:,1:], axis=0))
        cov.append(np.linalg.inv(np.cov(features_list[:,1:].T)))
    np.save("mean", mean)
    np.save("inv_cov", cov)
    # generate thresholds for cosine and mahalanobis
    tbar = tqdm.tqdm(val_dataloader, desc='\r') 

    with torch.no_grad():
        for i, img_data_list in enumerate(tbar):
            Fundus_img = img_data_list[0].to(device)
            cls_label = img_data_list[1].long().to(device)
            pred = model_features.forward(Fundus_img)
            pred_cls = model.forward(Fundus_img)
            
            # REVISAR, possible error al iterar.
            #distance = [np.min(cosine(i,j)) for i, j in zip(pred.cpu().numpy(), mean)]
            # ESTO ES LO QUE FALLA, REVISAR A FONDO MÉTODO, MEAN DE FEATURES FUNCIONA, Y SISTEMA DE MARCAR RELIABLE TAMBIEN
            # LA UNICA POSIBILIDAD ES FALLO AL CALCULAR THRESHOLD
            if Type:
                distance = []
                for i, p in enumerate(pred.cpu().numpy()):
                    distance.append(cosine(p, mean[cls_label[i]]))
                
                print(distance)
                # COMPARAR AMB PREDICCIONS COSINUS I NO DEL MODEL??              
                un_gt = 1 - torch.eq(pred_cls.argmax(dim=-1), cls_label).float()
                data_bach = pred.size(0)
                for idx in range(data_bach):
                    u_list.append(distance[idx])
                    u_label_list.append(un_gt.cpu()[idx].numpy())
            # mahalanobis
            else:
                distance = []
                for i, p in enumerate(pred.cpu().numpy()):
                    distance.append(mahalanobis(p, mean[cls_label[i]], cov[cls_label[i]]))
                    
                # COMPARAR AMB PREDICCIONS COSINUS I NO DEL MODEL??
                un_gt = 1 - torch.eq(pred_cls.argmax(dim=-1), cls_label).float()
                data_bach = pred.size(0)
                for idx in range(data_bach):
                    u_list.append(distance[idx])
                    u_label_list.append(un_gt.cpu()[idx].numpy())
        #print("Mean features by class",features_list)

    return u_list, u_label_list

def val(val_dataloader, model, args, mode, device):

    print('\n')
    print('====== Start {} ======!'.format(mode))
    model.eval()


    u_list = []
    u_label_list = []

    tbar = tqdm.tqdm(val_dataloader, desc='\r') 

    with torch.no_grad():
        for i, img_data_list in enumerate(tbar):
            Fundus_img = img_data_list[0].to(device)
            cls_label = img_data_list[1].long().to(device)
            pred = model.forward(Fundus_img)
            evidences = [F.softplus(pred)]

            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))

            u = args.num_classes / S
            print(u, u.shape)
            un_gt = 1 - torch.eq(b.argmax(dim=-1), cls_label).float()

            data_bach = pred.size(0)
            for idx in range(data_bach):
                u_list.append(u.cpu()[idx].numpy())
                u_label_list.append(un_gt.cpu()[idx].numpy())
    return u_list, u_label_list


def main(args=None):
    args.net_work = "ResUnNet50"
    args.trained_model_path = './Trained/Model_Kermany/model_Test_007_Val_0.994825_0.970634_Test_0.995626_0.969628.pth.tar'
    # bulid model
    device = torch.device('cuda:{}'.format(args.cuda))
    args.device = device
    model = net_builder(args.net_work, args.num_classes).to(device)

    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    # load trained model for test
    print("=> loading trained model '{}'".format(args.trained_model_path))
    checkpoint = torch.load(
        args.trained_model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print('Done!')

    torch.manual_seed(0)
    """
    split_ratio = 0.8
    data = Folders_dataset(path=args.root)
    train_data, val_data = torch.utils.data.random_split(data, [int(len(data)*split_ratio), len(data)-int(len(data)*split_ratio)])
    test_data, val_data = torch.utils.data.random_split(val_data, [int(len(val_data)*0.2), len(val_data)-int(len(val_data)*0.2)])
    train_sampler = class_sampler(data, train_data)
    """
    train_data = Folders_dataset(path=args.root+"/train")
    test_data = Folders_dataset(path=args.root+"/test")
    val_data = Folders_dataset(path=args.root+"/val")
    #del train_data, val_data, test_data, data
    train_loader = DataLoader(train_data,
        batch_size=args.batch_size, pin_memory=True)
    val_loader = DataLoader(val_data,
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data,
        batch_size=args.batch_size, shuffle=True, pin_memory=True)

    u_list, u_label_list = val_cosine(train_loader, val_loader, model, args, mode="Validation", device=device, Type=False) 
    #u_list, u_label_list = val(val_loader, model, args, mode="Validation", device=device) 

    fpr_Pri, tpr_Pri, thresh = metrics.roc_curve(u_label_list, u_list)
    max_j = max(zip(fpr_Pri, tpr_Pri), key=lambda x: 2*x[1] - x[0])
    pred_thresh = thresh[list(zip(fpr_Pri, tpr_Pri)).index(max_j)]
    print("opt_pred ===== {}".format(pred_thresh))




if __name__ == '__main__':
    args = DefaultConfig()

    main(args=args)