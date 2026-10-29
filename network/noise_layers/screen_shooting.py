import math
import random

import kornia
import torch
import torch.nn as nn
from kornia.geometry.transform import get_perspective_transform
from kornia.geometry.transform import warp_perspective

def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
	"""Computes affine matrix for translation."""
	matrix: torch.Tensor = torch.eye(
		3, device=translation.device, dtype=translation.dtype)
	matrix = matrix.repeat(translation.shape[0], 1, 1)

	dx, dy = torch.chunk(translation, chunks=2, dim=-1)
	matrix[..., 0, 2:3] += dx
	matrix[..., 1, 2:3] += dy
	return matrix

def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
	"""Computes the center of tensor plane for (H, W), (C, H, W) and (B, C, H, W)."""
	assert 2 <= len(tensor.shape) <= 4, f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}."
	height, width = tensor.shape[-2:]
	center_x: float = float(width - 1) / 2
	center_y: float = float(height - 1) / 2
	center: torch.Tensor = torch.tensor(
		[center_x, center_y],
		device=tensor.device, dtype=tensor.dtype)
	return center

def _compute_scaling_matrix(scale: torch.Tensor,
						center: torch.Tensor) -> torch.Tensor:
	"""Computes affine matrix for scaling."""
	angle: torch.Tensor = torch.zeros(scale.shape[0])
	matrix: torch.Tensor = kornia.get_rotation_matrix2d(center, angle, scale)
	return matrix

def _compute_rotation_matrix(angle: torch.Tensor,
							center: torch.Tensor) -> torch.Tensor:
	"""Computes a pure affine rotation matrix."""
	scale: torch.Tensor = torch.ones((angle.shape[0], 2))
	matrix: torch.Tensor = kornia.get_rotation_matrix2d(center, angle, scale)
	return matrix

def translate(image, device, d=8):
	c = image.shape[0]
	h = image.shape[-2]
	w = image.shape[-1]
	trans = torch.ones(c, 2)
	for i in range(c):
		dx = random.uniform(-d, d)
		dy = random.uniform(-d, d)

		trans[i, :] = torch.tensor([
			[dx, dy],
		])
	translation_matrix: torch.Tensor = _compute_translation_matrix(trans)
	matrix = translation_matrix[..., :2, :3]

	is_unbatched: bool = image.ndimension() == 3
	if is_unbatched:
		image = torch.unsqueeze(image, dim=0)

	matrix = matrix.expand(image.shape[0], -1, -1).to(device)

	data_warp: torch.Tensor = kornia.warp_affine(image, matrix, dsize=(h, w), padding_mode='border').to(device)

	if is_unbatched:
		data_warp = torch.squeeze(data_warp, dim=0)

	return data_warp

def rotate(image, device, d=8):
	c = image.shape[0]
	h = image.shape[-2]
	w = image.shape[-1]
	angle = torch.ones(c)
	center = torch.ones(c, 2)
	for i in range(c):
		an = random.uniform(-d, d)
		angle[i] = torch.tensor([an])
		center[i, :] = torch.tensor([[h / 2 - 1, w / 2 - 1], ])

	if center is None:
		center = _compute_tensor_center(image)

	angle = angle.expand(image.shape[0])
	center = center.expand(image.shape[0], -1)
	rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)

	matrix = rotation_matrix[..., :2, :3]

	is_unbatched: bool = image.ndimension() == 3
	if is_unbatched:
		image = torch.unsqueeze(image, dim=0)

	matrix = matrix.expand(image.shape[0], -1, -1).to(device)

	data_warp: torch.Tensor = kornia.warp_affine(image, matrix, dsize=(h, w), padding_mode='border').to(device)

	if is_unbatched:
		data_warp = torch.squeeze(data_warp, dim=0)

	return data_warp

def perspective(image, device, d=8):

	c = image.shape[0]
	h = image.shape[2]
	w = image.shape[3]
	work_dtype = image.dtype if image.is_floating_point() else torch.float32
	points_src = torch.ones(c, 4, 2, device=image.device, dtype=work_dtype)
	points_dst = torch.ones(c, 4, 2, device=image.device, dtype=work_dtype)
	for i in range(c):
		points_src[i, :, :] = torch.tensor([[
			[0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.],
		]], device=image.device, dtype=work_dtype)

		tl_x = random.uniform(-d, d)
		tl_y = random.uniform(-d, d)
		bl_x = random.uniform(-d, d)
		bl_y = random.uniform(-d, d)
		tr_x = random.uniform(-d, d)
		tr_y = random.uniform(-d, d)
		br_x = random.uniform(-d, d)
		br_y = random.uniform(-d, d)

		points_dst[i, :, :] = torch.tensor([[
			[tl_x, tl_y],
			[tr_x + (w - 1.), tr_y],
			[br_x + (w - 1.), br_y + (h - 1.)],
			[bl_x, bl_y + (h - 1.)],
		]], device=image.device, dtype=work_dtype)

	M: torch.tensor = get_perspective_transform(points_src, points_dst).to(image.device)

	data_warp: torch.tensor = warp_perspective(
		image.to(dtype=work_dtype), M, dsize=(h, w), padding_mode='border').to(device)

	return data_warp

def Light_Distortion_torch(c, embed_image):

	if not isinstance(embed_image, torch.Tensor):
		raise ValueError("embed_image must be a torch.Tensor")

	device = embed_image.device
	batch_size, channels, height, width = embed_image.size()

	mask_2d = torch.zeros(height, width, device=embed_image.device)
	a = (0.7 + torch.rand(1, device=device) * 0.2).item()
	b = (1.1 + torch.rand(1, device=device) * 0.2).item()
	if c == 0:
		for i in range(embed_image.shape[2]):
			mask_2d[i, :] = -((b - a) / (height - 1)) * (i - width) + a
		direction = torch.randint(1, 5, (1,), device=device).item()
		if direction == 1:
			O = mask_2d
		elif direction == 2:
			O = torch.rot90(mask_2d, 1, (0, 1))
		elif direction == 3:
			O = torch.rot90(mask_2d, 2, (0, 1))
		elif direction == 4:
			O = torch.rot90(mask_2d, 3, (0, 1))

	else:
		x = torch.randint(0, width, (1,), device=device)
		y = torch.randint(0, height, (1,), device=device)

		max_len = max(torch.sqrt(x ** 2 + y ** 2), torch.sqrt((x - 255) ** 2 + y ** 2),
					  torch.sqrt(x ** 2 + (y - 255) ** 2),
					  torch.sqrt((x - width) ** 2 + (y - height) ** 2)
					  ).item()

		y_coords = torch.arange(height, device=device).float()
		x_coords = torch.arange(width, device=device).float()
		Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')

		distances = torch.sqrt((X - x) ** 2 + (Y - y) ** 2)
		mask = (distances / max_len * (a - b) + b).unsqueeze(0).unsqueeze(0).expand(batch_size, channels, height,
																width).to(device)

		O = mask

	if O.dim() == 2:
		O = O.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, O.size(0), O.size(1)).contiguous()

	if O.size(2) != height or O.size(3) != width:
		O = torch.nn.functional.interpolate(O, size=(height, width), mode='bilinear', align_corners=False)

	return O

def MoireGen_torch(p_size, theta, center_x, center_y):
	theta = theta.to(torch.float32)
	center_x = center_x.to(torch.float32)
	center_y = center_y.to(torch.float32)

	M = torch.zeros((p_size, p_size), dtype=torch.float32)

	x_coords = torch.arange(p_size, dtype=torch.float32) + 1
	y_coords = torch.arange(p_size, dtype=torch.float32) + 1
	x_grid, y_grid = torch.meshgrid(x_coords, y_coords)

	x_grid = x_grid.to(theta.device)
	y_grid = y_grid.to(theta.device)

	if center_x.dim() == 0:
		center_x = center_x.unsqueeze(0).unsqueeze(0).expand_as(x_grid)
	if center_y.dim() == 0:
		center_y = center_y.unsqueeze(0).unsqueeze(0).expand_as(y_grid)

	dist_from_center = torch.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
	z1 = 0.5 + 0.5 * torch.cos(2 * math.pi * dist_from_center)

	theta_rad = theta / 180 * math.pi
	cos_theta = torch.cos(theta_rad)
	sin_theta = torch.sin(theta_rad)

	if theta.dim() == 0:
		cos_theta = cos_theta.unsqueeze(0).unsqueeze(0).expand_as(x_grid)
		sin_theta = sin_theta.unsqueeze(0).unsqueeze(0).expand_as(y_grid)

	z2_x = cos_theta * x_grid
	z2_y = sin_theta * y_grid
	z2 = 0.5 + 0.5 * torch.cos(2 * math.pi * (z2_x + z2_y))

	M = torch.min(z1, z2)

	M = (M + 1) / 2

	return M

def Moire_Distortion_torch(embed_image):
	if not isinstance(embed_image, torch.Tensor):
		raise ValueError("embed_image must be a torch.Tensor")

	device = embed_image.device
	batch_size, _, h, w = embed_image.size()
	Z = torch.zeros(batch_size, 3, h, w, device=device)
	p_size = int(max(h, w))

	for i in range(3):
		theta = torch.randint(0, 180, (1,), device=device)
		center_x = torch.rand(1, device=device) * p_size
		center_y = torch.rand(1, device=device) * p_size

		M = MoireGen_torch(p_size, theta, center_x, center_y)

		if M.dim() != 2:
			raise ValueError("MoireGen_torch must return a 2D tensor")

		M = M[:h, :w].unsqueeze(0).expand(batch_size, h, w).to(device)

		Z[:, i, :, :] = M

	return Z

class ScreenShooting(nn.Module):

	def __init__(self, distance=2):
		super(ScreenShooting, self).__init__()
		self.distance = distance

	def forward(self, image_and_cover):
		image, *_ = image_and_cover
		embed_image = image

		device = embed_image.device

		noised_image = perspective(embed_image, device, self.distance)

		c = torch.randint(0, 2, (1,), device=device)
		L = Light_Distortion_torch(c, embed_image)

		Z = Moire_Distortion_torch(embed_image) * 2 - 1

		noised_image = noised_image * L * 0.85 + Z * 0.15

		noised_image = noised_image + 0.001 ** 0.5 * torch.randn(noised_image.size(), dtype=torch.float).to(device)

		return noised_image
