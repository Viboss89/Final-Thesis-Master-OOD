def net_builder(name,num_classes=9):
    if name == 'ResUnNet50':
        from models.ResNet.ResUnNet import ResUnNet
        net= ResUnNet(num_classes=num_classes)
        print(net)
    elif name== 'ViT':
        from models.ResNet.ResUnNet import ViT
        net = ViT(num_classes=num_classes)
        print(net)
    else:
        raise NameError("Unknow Model Name!")
    return net
