import torch
import torch.nn as nn
import numpy as np

class RP(nn.Module):

    def __init__(self, npp):
        super(RP, self).__init__()
        self.npp = npp

    def reduce_precision_py(self, image, npp):
        """
        Reduce the precision of image, the numpy version.
        :param x: a float tensor, which has been scaled to [0, 1].
        :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
        :return: a tensor representing image(s) with lower precision.
        """

        npp_int = npp - 1

        image = image.to('cuda')

        x_int = (image * npp_int).round()

        x_float = x_int / npp_int
        return x_float

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        image = self.reduce_precision_py(image, self.npp)
        return image
