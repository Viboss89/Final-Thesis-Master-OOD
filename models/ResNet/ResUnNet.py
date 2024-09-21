import torch
import torchvision
import torch.nn as nn

class ResUnNet(nn.Module):
    def __init__(self,num_classes=1):
        print("=================== ResUnNet ===================")
        super(ResUnNet, self).__init__()
        resnet_img = torchvision.models.resnet50(pretrained=True)  # pretrained ImageNet ResNet-34
        modules_img = list(resnet_img.children())[:-2]
        self.resnet_img = nn.Sequential(*modules_img)

        self.avgpool_fun = nn.AdaptiveAvgPool2d((1,1))  #
        self.affine_classifier = nn.Linear(2048, num_classes)
    def forward(self, image):

        out_img = self.resnet_img(image)
        avg_feature = self.avgpool_fun(out_img)
        avg_feature = torch.flatten(avg_feature, 1)
        result = self.affine_classifier(avg_feature)
        return result
        
        
class ViT(nn.Module):
    def __init__(self,num_classes=1):
        print("=================== ViT ===================")
        super(ViT, self).__init__()
        self.resnet_img = torchvision.models.vit_b_16(pretrained=torchvision.models.ViT_B_16_Weights.DEFAULT)  # pretrained ImageNet ResNet-34
        print(self.resnet_img.heads.head)
        self.resnet_img.heads.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)
        
    def forward(self, image):
        result = self.resnet_img(image)
        return result
