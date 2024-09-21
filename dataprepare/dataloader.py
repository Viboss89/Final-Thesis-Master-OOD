# --coding:utf-8--
from torch.utils.data import Dataset
from torchvision import transforms as T 
from PIL import Image
import numpy as np
from torch.utils.data import WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torch import from_numpy
import pandas as pd
from itertools import islice
import matplotlib.pyplot as plt

class DatasetCFP(Dataset):
    def __init__(self,root,data_file,mode = 'train'):
        self.data_list = self.get_files(root,data_file=data_file)
        if mode == 'train':
            self.transforms= T.Compose([
                T.Resize((448,448)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],
                            std = [0.299,0.224,0.225])
            ])
        else:

            self.transforms = T.Compose([
                T.Resize((448,448)),
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],
                            std = [0.299,0.224,0.225])
            ])

    def get_files(self,root,data_file):
        csv_labels = pd.read_csv(data_file, header=None).to_numpy()
        return csv_labels



    def __getitem__(self,index):
        image_file, label = self.data_list[index]
        img = Image.open(image_file).convert("RGB")
        img_tensor = self.transforms(img)

        return img_tensor, label

    def __len__(self):
        return len(self.data_list)
    
def Folders_dataset(path, mode='train'):
    
    if mode == 'train':
            transforms= T.Compose([
                T.Resize((448,448)),           #Resnet
                #T.Resize((224,224)),            #ViT
                T.RandomHorizontalFlip(),
                #T.GaussianBlur(kernel_size=7,sigma=2.0),   activate to test OOD detection
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],
                            std = [0.299,0.224,0.225])
            ])
    else:

            transforms = T.Compose([
                T.Resize((448,448)),           #Resnet 
                #T.Resize((224,224)),            #ViT
                T.ToTensor(),
                T.Normalize(mean = [0.485,0.456,0.406],
                            std = [0.299,0.224,0.225])
            ])
    dataset = ImageFolder(path, transform=transforms)

    return dataset


def class_sampler(data, train_data, k=1):
    """
    Sampler to balance dataset when it is imbalanced
    """
    
    y_train_indices = train_data.indices
    y_train = [data.targets[i] for i in y_train_indices]
    class_sample_count = np.array(
                            [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])

    weight = 1./class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), k*len(samples_weight))
    return sampler
