from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.a_path = join(image_dir, "images")
        self.b_path = join(image_dir, "masks")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        images = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        masks = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
#         a = a.resize((286, 286), Image.BICUBIC)
#         b = b.resize((286, 286), Image.BICUBIC)
        images = transforms.ToTensor()(images)
        masks = transforms.ToTensor()(masks)
#         w_offset = random.randint(0, max(0, 286 - 256 - 1))
#         h_offset = random.randint(0, max(0, 286 - 256 - 1))
#     
#         a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
#         b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
#   
#         a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
#         b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)
# 
#        'mirror the input images and masks'
#         if random.random() < 0.5:
#             idx = [i for i in range(a.size(2) - 1, -1, -1)]
#             idx = torch.LongTensor(idx)
#             a = a.index_select(2, idx)
#             b = b.index_select(2, idx)

        return images, masks

    def __len__(self):
        return len(self.image_filenames)
