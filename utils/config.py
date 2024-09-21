# -*- coding: utf-8 -*-
class DefaultConfig(object):
    net_work = 'ViT'
    num_classes = 4
    num_epochs = 6
    batch_size = 5
    validation_step = 2
    root = "../data/Kermany"
    train_file = "../ZhangLabData/train.csv"
    val_file ="../ZhangLabData/val.csv"
    test_file = "../ZhangLabData/test.csv"
    lr = 1e-4
    lr_mode = 'poly'
    momentum = 0.9
    weight_decay = 1e-4
    #save_model_path = './Model_Saved'.format(net_work,lr)
    save_model_path = './Trained/full_dataset_Transf_ood'
    log_dirs = './Logs_Adam_0304'
    pretrained = False
    pretrained_model_path = None
    cuda = 0
    num_workers = 4
    use_gpu = True
    trained_model_path = ''
    predict_fold = 'predict_mask'
