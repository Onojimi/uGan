from __future__ import print_function
import argparse
import os
from math import log10
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from models.model_gen import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_val_set
from combineloss import mixloss
from eval import eval_net
from tensorboardX import SummaryWriter

def compute_iou(true, pred):
    true_mask = np.asanyarray(true.cpu(), dtype = np.bool)
    pred_mask = np.asanyarray(pred, dtype = np.bool)
    union = np.sum(np.logical_or(true_mask, pred_mask))
    intersection = np.sum(np.logical_and(true_mask, pred_mask))
#    print(union, intersection)
    iou = intersection/union
    return iou

parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--val_batch_size', type=int, default=1, help='val batch size')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--load_pretrain', action='store_true', help='use cuda?')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--loss', type= str, default = 'L1', help='type of loss function')
parser.add_argument('--debug', type=int, default = 0, help='type of loss function')

opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(root_path + opt.dataset)
val_set = get_val_set(root_path + opt.dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
val_data_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.val_batch_size, shuffle=False)

device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Building models')
net_g = define_G(opt.input_nc, opt.output_nc,'batch','normal', 0.02, gpu_ids=opt.gpu_ids)
net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_ids=opt.gpu_ids)

optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.load_pretrain:
    print('Model loaded from {}'.format(opt.load_pretrain))
    model_dict = net_g.state_dict()
    pretrained_dict = torch.load('CP50.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
    model_dict.update(pretrained_dict)
    net_g.load_state_dict(model_dict)
    train_params = []
    for k, v in net_g.named_parameters():
        train_params.append(k)
        pref = k[:12]
        if pref == 'module.conv1' or pref == 'module.conv2' :
            v.requires_grad=False
            train_params.remove(k)
        
    optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, net_g.parameters()),
                             lr=opt.lr, 
                             betas=(opt.beta1, 0.999)) 

criterionGAN = GANLoss().to(device)
criterionMSE = nn.MSELoss().to(device)

if opt.loss == 'mixloss':
    print('==>using BCE+Dice as loss function')
    criterionGen = mixloss().to(device)
else:
    criterionGen = nn.L1Loss().to(device)
    print('==>using L1 loss as loss function')
# setup optimizer

net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

writer = SummaryWriter(log_dir='../../log/sn1', comment='unet')

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        net_g.train()
        
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b = net_g(real_a)

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_gen = criterionGen(fake_b, real_b) * opt.lamb
        
        loss_g = loss_g_gan + loss_gen
        
        loss_g.backward()

        optimizer_g.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # test
#     avg_psnr = 0
#     for batch in val_data_loader:
#         input, target = batch[0].to(device), batch[1].to(device)
#         prediction = net_g(input)
#         mse = criterionMSE(prediction, target)
#         psnr = 10 * log10(1 / mse.item())
#         avg_psnr += psnr
    avg_ls, avg_iou = eval_net(val_data_loader, net_g, device)
    print("===>Generator: Avg. iou: {:.4f} , Avg. loss:{:.4f}".format(avg_iou, avg_ls))
    writer.add_scalar('train/loss', loss_g.item(), epoch)
    writer.add_scalar('val/loss', avg_ls, epoch )
    writer.add_scalar('val/IoU', avg_iou, epoch )
    #checkpoint
    if epoch % 50 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        torch.save(net_g, net_g_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))