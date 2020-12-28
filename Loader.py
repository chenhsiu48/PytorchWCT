from PIL import Image
import skimage.transform
import torchvision.transforms as transforms
import torch.utils.data as data
from os import listdir
import os
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')


class DatasetOne(data.Dataset):
    def __init__(self, contentPath, stylePath, fineSize=0):
        super(DatasetOne, self).__init__()
        self.contentPath = contentPath
        self.stylePath = stylePath
        self.image_list = [f'{os.path.basename(contentPath).split(".")[0]}-{os.path.basename(stylePath).split(".")[0]}.png']
        self.fineSize = fineSize

    def __getitem__(self, index):
        contentImg = default_loader(self.contentPath)
        styleImg = default_loader(self.stylePath)

        contentImg = transforms.Compose([RescaleNew(self.fineSize), CropModulus(16), transforms.ToTensor()])(contentImg)
        styleImg = transforms.Compose([RescaleNew((contentImg.shape[1], contentImg.shape[2])), transforms.ToTensor()])(styleImg)
        return contentImg, styleImg, self.image_list[index]

    def __len__(self):
        return len(self.image_list)


class RescaleNew(object):
    def __init__(self, target_size, scaling='bigger_side', interpolation=Image.BILINEAR):
        self.interpolation = interpolation
        self.scaling = scaling
        self.target_size = target_size

    def target_shape(self, H, W):
        if (self.scaling == 'bigger_side' and H > W) or (self.scaling == 'smaller_side' and H < W):
            Wnew = int(np.round(W / H * self.target_size))
            Hnew = self.target_size
        else:
            Wnew = self.target_size
            Hnew = int(np.round(H / W * self.target_size))
        return Hnew, Wnew

    def __call__(self, image):
        W, H = image.size
        if self.target_size == 0:
            Hnew, Wnew = H, W
        elif len(self.target_size) == 2:
            Hnew, Wnew = self.target_size
        else:
            Hnew, Wnew = self.target_shape(H, W)

        assert min(Hnew, Wnew) >= 224, f'invalid target_size {Hnew, Wnew}'

        return image.resize((Wnew, Hnew), resample=self.interpolation)


class CropModulus(object):

    def __init__(self, crop_modulus, mode='PIL'):
        assert mode in ['PIL', 'HWC', 'CHW']
        self.mode = mode
        self.crop_modulus = crop_modulus

    def __call__(self, im):
        if self.mode == 'PIL':
            W, H = im.size
        elif self.mode == 'HWC':
            H, W = im.shape[:2]
        elif self.mode == 'HWC':
            H, W = im.shape[1:3]
        Hmod = H - H % self.crop_modulus
        Wmod = W - W % self.crop_modulus
        border_x = (W - Wmod) // 2
        border_y = (H - Hmod) // 2
        end_x = border_x + Wmod
        end_y = border_y + Hmod
        crop_box = (border_x, border_y, end_x, end_y)
        if self.mode == 'PIL':
            return im.crop(crop_box)
        elif self.mode == 'HWC':
            return im[border_y:end_y, border_x:end_x, :]
        else:  # self.mode == 'HWC':
            return im[: border_y:end_y, border_x:end_x]
