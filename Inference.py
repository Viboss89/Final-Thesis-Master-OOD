import torch
import tqdm
from torch.utils.data import DataLoader
from utils.config import DefaultConfig
from models.net_builder import net_builder
from dataprepare.dataloader import DatasetCFP, class_sampler, Folders_dataset
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine, mahalanobis
from sklearn.metrics.pairwise import cosine_similarity

import csv

def val(val_dataloader, model, args, Normal_average=0.0,device=None):


    modes = ["dirichlet", "cosine", "mahalanobis"]
    mode = modes[2]
    print('\n')
    model.eval()
    model_features = net_builder(args.net_work, args.num_classes).to(device)

    print('Model have been loaded!,you chose the ' + args.net_work + '!')
    # load trained model for test
    print("=> loading trained model '{}'".format(args.trained_model_path))
    checkpoint = torch.load(
        args.trained_model_path)
    model_features.load_state_dict(checkpoint['state_dict'])
    print('Done!')
    model_features.affine_classifier = torch.nn.Sequential()
    model_features.eval()
    tbar = tqdm.tqdm(val_dataloader, desc='\r')
    All_Infor = []


    mean = np.load("mean.npy", allow_pickle=True)
    inv_cov = np.load("inv_cov.npy", allow_pickle=True)
    with torch.no_grad():
        for i, img_data_list in enumerate(tbar):
            Fundus_img = img_data_list[0].to(device)
            image_files = img_data_list[1]
            pred = model.forward(Fundus_img)
            evidences = [F.softplus(pred)]

            alpha = dict()
            alpha[0] = evidences[0] + 1

            S = torch.sum(alpha[0], dim=1, keepdim=True)
            E = alpha[0] - 1
            b = E / (S.expand(E.shape))

            u = args.num_classes / S

            batch = b.shape[0]

            pred_con = pred.argmax(dim=-1)


            # check dirichlet
            if mode=="dirichlet":
                for idx_bs_u in range(batch):

                    if u[idx_bs_u] >= Normal_average:
                        """
                        All_Infor.append([
                            '/'.join(image_files[idx_bs_u].split('/')[-2:]),
                            pred_con.cpu().detach().float().numpy()[idx_bs_u],
                            "Unreliable"
                        ]
                        )
                    else:
                        All_Infor.append([
                            '/'.join(image_files[idx_bs_u].split('/')[-2:]),
                            pred_con.cpu().detach().float().numpy()[idx_bs_u],
                            "Reliable"
                        ]
                        )
                        """
                        dirichlet_reliability = "Reliable"

                    else:  
                        dirichlet_reliability = "Unreliable"

                    All_Infor.append([str(image_files[idx_bs_u].cpu().detach().int().numpy()), str(pred_con.cpu().detach().int().numpy()[idx_bs_u]), dirichlet_reliability])
            
            if mode=="cosine" or "mahalanobis":
                features = model_features.forward(Fundus_img)
                for idx_bs_u , (feature, pred) in enumerate(zip(features.cpu().numpy(), pred_con)):
                    cosines = [] 
                    mahalanobis_d = []
                    pred = pred.cpu().item()
                    for categ in range(args.num_classes):
                        cosines.append(cosine(feature, mean[categ]))
                        mahalanobis_d.append(mahalanobis(feature, mean[categ], inv_cov[categ]))
                    
                    print(cosines[pred], cosines, cosines[pred]<=Normal_average)
                    if mode=="cosine" and cosines[pred]<=Normal_average:
                        print("Hola")
                        reliability = "Reliable"
                    elif mode=="mahalanobis" and mahalanobis_d[pred]<=Normal_average:
                        reliability ="Reliable"
                    else:
                        reliability ="Unreliable"
                    print(reliability)
                    All_Infor.append([str(image_files[idx_bs_u].cpu().detach().int().numpy()), str(pred_con.cpu().detach().int().numpy()[idx_bs_u]), reliability])


    return All_Infor


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


    Thres = 0.076
    Results_Heads = ["Imagefiles", 'Prediction results','Reliability']



    args.root = "../data/Kermany"
    csv_file = "./Datasets/Pred_test_standard.csv"

    torch.manual_seed(0)
    split_ratio = 0.8
    """
    data = Folders_dataset(path=args.root, mode='test')
    train_data, test_data = torch.utils.data.random_split(data, [int(len(data)*split_ratio), len(data)-int(len(data)*split_ratio)])
    #test_data, val_data = torch.utils.data.random_split(val_data, [int(len(val_data)*0.2), len(val_data)-int(len(val_data)*0.2)])
    """
    train_data = Folders_dataset(path=args.root+"/train", mode="")
    test_data = Folders_dataset(path=args.root+"/test", mode="")
    val_data = Folders_dataset(path=args.root+"/val", mode="")
    #train_sampler = class_sampler(data, train_data)
    #del train_data, val_data, test_data, data
    train_loader = DataLoader(train_data,
        batch_size=args.batch_size, pin_memory=True)
    val_loader = DataLoader(val_data,
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data,
        batch_size=args.batch_size, shuffle=True, pin_memory=True)

    Results_Contents = val(test_loader, model, args,
        Normal_average=Thres, device=device)
    with open(
            "test_maha.csv", 'a',     #modify name to generate reports with csv format
            newline='') as f:
        writer = csv.writer(f)
        writer.writerow(Results_Heads)
        writer.writerows(
            Results_Contents
        )


if __name__ == '__main__':
    args = DefaultConfig()

    main(args=args)