from __future__ import print_function

import os
import sys
import argparse
import time
import math
from PIL import Image
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from losses import SupConLoss
from dataloader_suc import *

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='path',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, default='(0.4089, 0.4217, 0.3537)',help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, default='(0.1712, 0.1477, 0.1496)',help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default='./datasets/e_shanghai', help='path to custom dataset')
    parser.add_argument('--size', type=int, default=64, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.Resize(size=opt.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = MyDataset(
            #'../../image_name_POI_level_1.csv',
            './corr_e_shanghai/image_name_POI_sh.csv',
            #'corr_file/corr_POI_cate_1st_duiqi_euclid_POI10.txt',
            './corr_e_shanghai/corr_POI_1st_sh.txt',
        './corr_e_shanghai/corr_human_sh.txt',
        './corr_e_shanghai/corr_Geo_sh.txt',
            #'corr_POI_cate_1st_duiqi_euclid_certain_POI.txt',
            #'corr_max.txt',
            #'correspond_last2000.txt',
            './datasets/e_shanghai/',
            transform=train_transform,
        )


    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False,drop_last=False,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader

def set_model(opt):
    model = SupConResNet(name=opt.model)
    return model

def inference(loader, con_model, device):
    feature_vector = []
    for step, (x, y, y1, y2) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h = con_model(x)

        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector


def get_features(simclr_model, train_loader, device):
    train_X= inference(train_loader, simclr_model, device)
    print(train_X.shape)

    return train_X


if __name__ == "__main__":
    opt = parse_option()
    train_loader = set_loader(opt)
    model = SupConResNet(name=opt.model)


    model_fp = os.path.join('./save/', "final_dict_eshanghai.pth")
    model.load_state_dict(torch.load(model_fp, map_location="cuda"))
    simclr_model = model.to('cuda')
    simclr_model.eval()

    print("### Creating features from pre-trained context model ###")
    train_X= get_features(
        simclr_model, train_loader, 'cuda'
    )

    np.savetxt('region_embeding_sh.txt', (train_X), fmt='%f')  # save the image embeddings


