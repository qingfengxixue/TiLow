import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_locator_unit():

	d = 8
	sigma = 0.4
	beta = 0.4
	center_x = 3.5
	center_y = 3.5
	grid = torch.arange(d, dtype=torch.float32)
	yy, xx = torch.meshgrid(grid, grid, indexing="ij")
	D = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
	m_sc = torch.where(D < 3.2, torch.zeros_like(D), torch.full_like(D, beta))
	pos = 0.5 + torch.stack([m_sc, torch.zeros_like(m_sc), m_sc], dim=0)
	neg = 0.5 + torch.stack([-m_sc, torch.zeros_like(m_sc), -m_sc], dim=0)
	unit = torch.where(D < 4, pos, neg)
	unit = unit.clamp(0.0, 1.0)
	return unit

class TemplateWatermark(nn.Module):
	"""
	根据 message 生成小模板并平铺到整张图片，再与图片融合。
	输出 S 与模板/掩码，便于调试。
	"""
	def __init__(self, d=8, alpha=0.15, center_peak=True, tile_size=64, unit_size=8,
				 single_tile=False, random_position=False):
		super().__init__()
		self.d = d
		self.alpha = alpha
		self.center_peak = center_peak
		self.tile_size = tile_size
		self.unit_size = unit_size
		self.tile_units = tile_size // unit_size
		self.single_tile = single_tile
		self.random_position = random_position

	def _make_m_sc(self, device, dtype):
		grid = torch.arange(self.unit_size, device=device, dtype=dtype)
		yy, xx = torch.meshgrid(grid, grid, indexing="ij")
		center = (self.unit_size - 1) / 2.0
		dist = torch.sqrt((xx - center) ** 2 + (yy - center) ** 2)
		max_dist = dist.max().clamp(min=1.0)
		m_sc = dist / max_dist
		if self.center_peak:
			m_sc = 1.0 - m_sc
		return m_sc

	def _pad_or_trim_message(self, message, count):
		length = message.shape[1]
		if length < count:
			message = F.pad(message, (0, count - length))
		elif length > count:
			message = message[:, :count]
		return message

	def _locator_coords(self):
		return {
			(0, 0), (0, 1), (1, 0), (1, 1),
			(0, 6), (0, 7), (1, 6), (1, 7),
			(6, 0), (6, 1), (7, 0), (7, 1),
			(6, 6), (6, 7), (7, 6), (7, 7),
		}

	def _build_locator_tile_mask(self, device, dtype):
		locator = self._locator_coords()
		unit = generate_locator_unit().to(device=device, dtype=dtype)
		mag = (unit - 0.5).abs().mean(dim=0, keepdim=True)
		mag = (mag / 0.4).clamp(0.0, 1.0)
		mask = torch.zeros((1, self.tile_size, self.tile_size), device=device, dtype=dtype)
		for uy, ux in locator:
			y0 = uy * self.unit_size
			x0 = ux * self.unit_size
			mask[:, y0:y0 + self.unit_size, x0:x0 + self.unit_size] = mag
		return mask

	def build_locator_map(self, h, w, offsets, device, dtype):
		b = len(offsets)
		locator_tile = self._build_locator_tile_mask(device, dtype)
		locator_map = torch.zeros((b, 1, h, w), device=device, dtype=dtype)
		if self.single_tile:
			for i, (top, left) in enumerate(offsets):
				locator_map[i, :, top:top + self.tile_size, left:left + self.tile_size] = locator_tile
		else:
			tile_h = int(math.ceil(h / self.tile_size))
			tile_w = int(math.ceil(w / self.tile_size))
			tiled = locator_tile.repeat(1, tile_h, tile_w)[:, :h, :w]
			locator_map[:] = tiled
		return locator_map

	def _message_positions(self):
		locator = self._locator_coords()
		positions = []
		for uy in range(self.tile_units):
			for ux in range(self.tile_units):
				if (uy, ux) in locator:
					continue
				positions.append((uy, ux))
		return positions

	def forward(self, image, message, alpha=None):
		if message.dtype != torch.float32:
			message = message.float()
		alpha = self.alpha if alpha is None else alpha
		b, _, h, w = image.shape

		positions = self._message_positions()
		message = self._pad_or_trim_message(message, len(positions))
		sign = message * 2 - 1

		m_sc = self._make_m_sc(image.device, image.dtype)
		m_sc = m_sc.view(1, self.unit_size, self.unit_size)
		locator_unit = generate_locator_unit().to(device=image.device, dtype=image.dtype)
		locator_unit = (locator_unit - 0.5) * 2.0

		m_s = torch.zeros((b, 3, self.tile_size, self.tile_size), device=image.device, dtype=image.dtype)
		locator = self._locator_coords()
		msg_idx = 0
		for uy in range(self.tile_units):
			for ux in range(self.tile_units):
				y0 = uy * self.unit_size
				x0 = ux * self.unit_size
				if (uy, ux) in locator:
					m_s[:, :, y0:y0 + self.unit_size, x0:x0 + self.unit_size] = locator_unit.unsqueeze(0)
				else:
					bit_sign = sign[:, msg_idx]
					msg_idx += 1
					unit = bit_sign[:, None, None] * m_sc
					red = unit
					green = torch.zeros_like(unit)
					blue = -unit
					unit_rgb = torch.stack([red, green, blue], dim=1)
					m_s[:, :, y0:y0 + self.unit_size, x0:x0 + self.unit_size] = unit_rgb

		offsets = []
		if self.single_tile:
			m = torch.zeros((b, 3, h, w), device=image.device, dtype=image.dtype)
			for i in range(b):
				if self.random_position:
					top = torch.randint(0, max(1, h - self.tile_size + 1), (1,), device=image.device).item()
					left = torch.randint(0, max(1, w - self.tile_size + 1), (1,), device=image.device).item()
				else:
					top, left = 0, 0
				m[i, :, top:top + self.tile_size, left:left + self.tile_size] = m_s[i]
				offsets.append((top, left))
		else:
			tile_h = int(math.ceil(h / self.tile_size))
			tile_w = int(math.ceil(w / self.tile_size))
			m = m_s.repeat(1, 1, tile_h, tile_w)[:, :, :h, :w]
			offsets = [(0, 0)] * b

		s = alpha * m + (1 - alpha) * image
		s = s.clamp(-1, 1)
		return s, m_s, m, offsets
