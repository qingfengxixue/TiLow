import os
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset

class Dataset(Dataset):

	def __init__(self, path, H=256, W=256):
		super(Dataset, self).__init__()
		self.H = H
		self.W = W
		self.path = path
		self.min_hw = 1
		self.list = []
		for name in os.listdir(path):
			file_path = os.path.join(path, name)
			if not os.path.isfile(file_path):
				continue
			try:
				with Image.open(file_path) as img:
					w, h = img.size
					if w > self.min_hw and h > self.min_hw:
						self.list.append(name)
			except Exception:
				continue

		if len(self.list) == 0:
			raise RuntimeError(f"No valid images found in dataset path: {path}")

		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])

	def transform_image(self, image):

		if image.size[0] < image.size[1] / 2 or image.size[1] < image.size[0] / 2:
			return None

		image = self.transform(image)

		return image

	def __getitem__(self, index):

		while True:
			index = index % len(self.list)
			image = Image.open(os.path.join(self.path, self.list[index])).convert("RGB")
			image = self.transform_image(image)
			if image is not None:
				return image

			index += 1

	def __len__(self):
		return len(self.list)
