from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

from utils import is_image_file, load_img, save_img
from eval import compute_iou

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)

net_g = torch.load(model_path).to(device)
net_g.eval()

image_dir = "dataset/{}/test/images/".format(opt.dataset)
mask_dir = "dataset/{}/test/masks/".format(opt.dataset)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                  ]

transform = transforms.Compose(transform_list)

iou = 0
ct = 0
for image_name in image_filenames:
    true_mask = load_img(mask_dir + image_name)
    true_mask = transform(true_mask)
    
    img = load_img(image_dir + image_name)
    img = transform(img)
    
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    
    iou += compute_iou(true_mask, out)
    out_img = out.detach().squeeze(0).cpu()
    ct +=1
    
    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.makedirs(os.path.join("result", opt.dataset))
    save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))

print("IoU on test set is {:.4f}".format(iou/ct))