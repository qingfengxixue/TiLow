import torch
import torch.nn as nn
from kornia.filters import GaussianBlur2d

class Bright(nn.Module):

    def RGBAlgorithm(self, rgb_img, value=0.5, basedOnCurrentValue=True):
        img = rgb_img * 1.0

        img_out = (img.clamp(-1, 1) + 1) * 255 / 2
        img = img_out

        if basedOnCurrentValue:

            if value >= 0:
                alpha = 1 - value
                alpha = 1 / alpha

            else:
                alpha = value + 1

            img_out[:, 0, :, :] = img[:, 0, :, :] * alpha
            img_out[:, 1, :, :] = img[:, 1, :, :] * alpha
            img_out[:, 2, :, :] = img[:, 2, :, :] * alpha

        else:
            alpha = value
            img_out[:, 0:1, :, :] = img[:, 0:1, :, :] + 255.0 * alpha
            img_out[:, 1:2, :, :] = img[:, 1:2, :, :] + 255.0 * alpha
            img_out[:, 2:3, :, :] = img[:, 2:3, :, :] + 255.0 * alpha

        img_out = img_out / 255.0

        mask_3 = img_out < 0
        mask_4 = img_out > 1
        img_out = img_out * (~mask_3 + 0)
        img_out = img_out * (~mask_4 + 0) + mask_4

        img_out = img_out * 2 - 1
        return img_out

    def __init__(self, Increment, add_gf=False, gf_sigma=0):
        super(Bright, self).__init__()
        self.value = Increment / 255.0
        self.basedOnCurrentValue = True
        if gf_sigma is not 0:
            self.add_gf = add_gf
            self.gf_sigma = gf_sigma
            self.gaussian_filter = GaussianBlur2d((3, 3), (gf_sigma, gf_sigma))

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        image = self.RGBAlgorithm(image, self.value, self.basedOnCurrentValue)
        if self.add_gf is True:
            image = self.gaussian_filter(image)
        return image
