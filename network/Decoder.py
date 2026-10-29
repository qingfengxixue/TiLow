from . import *

class ConvBlock(nn.Module):
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.block(x)

class ConvBNReLU(nn.Module):
	def __init__(self, in_ch, out_ch, k=3, p=1):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.block(x)

class LocatorUNetPP(nn.Module):
	"""
	阶段1定位器：UNet++（分割）
	输入: Ssub
	输出: lpred (logits)
	"""

	def __init__(self, in_ch=3, base_ch=32):
		super().__init__()
		self.pool = nn.MaxPool2d(2, 2)

		self.conv00 = ConvBlock(in_ch, base_ch)
		self.conv10 = ConvBlock(base_ch, base_ch * 2)
		self.conv20 = ConvBlock(base_ch * 2, base_ch * 4)
		self.conv30 = ConvBlock(base_ch * 4, base_ch * 8)

		self.conv01 = ConvBlock(base_ch + base_ch * 2, base_ch)
		self.conv11 = ConvBlock(base_ch * 2 + base_ch * 4, base_ch * 2)
		self.conv21 = ConvBlock(base_ch * 4 + base_ch * 8, base_ch * 4)

		self.conv02 = ConvBlock(base_ch * 2 + base_ch * 2, base_ch)
		self.conv12 = ConvBlock(base_ch * 4 + base_ch * 4, base_ch * 2)

		self.conv03 = ConvBlock(base_ch * 3 + base_ch * 2, base_ch)

		self.head = nn.Conv2d(base_ch, 1, kernel_size=1)

	@staticmethod
	def _up(x, size):
		return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

	def forward(self, x):
		x00 = self.conv00(x)
		x10 = self.conv10(self.pool(x00))
		x20 = self.conv20(self.pool(x10))
		x30 = self.conv30(self.pool(x20))

		x01 = self.conv01(torch.cat([x00, self._up(x10, x00.shape[2:])], dim=1))
		x11 = self.conv11(torch.cat([x10, self._up(x20, x10.shape[2:])], dim=1))
		x21 = self.conv21(torch.cat([x20, self._up(x30, x20.shape[2:])], dim=1))

		x02 = self.conv02(torch.cat([x00, x01, self._up(x11, x00.shape[2:])], dim=1))
		x12 = self.conv12(torch.cat([x10, x11, self._up(x21, x10.shape[2:])], dim=1))

		x03 = self.conv03(torch.cat([x00, x01, x02, self._up(x12, x00.shape[2:])], dim=1))
		return self.head(x03)

class WatermarkDecoderUNet3Plus(nn.Module):
	"""
	阶段2解码器：UNet3+（UNet+++）骨干 + 多层卷积映射到二值序列
	输入: Spred
	输出: wpred (logits, shape [B, message_length])
	"""

	def __init__(self, message_length, in_ch=3, base_ch=32, agg_ch=32):
		super().__init__()
		self.message_length = message_length

		self.enc1 = ConvBlock(in_ch, base_ch)
		self.enc2 = ConvBlock(base_ch, base_ch * 2)
		self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
		self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)

		self.pool = nn.MaxPool2d(2, 2)

		self.e1_d3 = ConvBNReLU(base_ch, agg_ch)
		self.e2_d3 = ConvBNReLU(base_ch * 2, agg_ch)
		self.e3_d3 = ConvBNReLU(base_ch * 4, agg_ch)
		self.e4_d3 = ConvBNReLU(base_ch * 8, agg_ch)
		self.d3_fuse = ConvBNReLU(agg_ch * 4, agg_ch * 4)

		self.e1_d2 = ConvBNReLU(base_ch, agg_ch)
		self.e2_d2 = ConvBNReLU(base_ch * 2, agg_ch)
		self.e3_d2 = ConvBNReLU(base_ch * 4, agg_ch)
		self.e4_d2 = ConvBNReLU(base_ch * 8, agg_ch)
		self.d3_d2 = ConvBNReLU(agg_ch * 4, agg_ch)
		self.d2_fuse = ConvBNReLU(agg_ch * 5, agg_ch * 4)

		self.e1_d1 = ConvBNReLU(base_ch, agg_ch)
		self.e2_d1 = ConvBNReLU(base_ch * 2, agg_ch)
		self.e3_d1 = ConvBNReLU(base_ch * 4, agg_ch)
		self.e4_d1 = ConvBNReLU(base_ch * 8, agg_ch)
		self.d3_d1 = ConvBNReLU(agg_ch * 4, agg_ch)
		self.d2_d1 = ConvBNReLU(agg_ch * 4, agg_ch)
		self.d1_fuse = ConvBNReLU(agg_ch * 6, agg_ch * 4)

		self.refine = nn.Sequential(
			ConvBNReLU(agg_ch * 4, agg_ch * 4),
			ConvBNReLU(agg_ch * 4, agg_ch * 2),
		)
		self.head = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Linear(agg_ch * 2, message_length),
		)

	@staticmethod
	def _resize(x, size):
		return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

	def forward(self, x):
		e1 = self.enc1(x)
		e2 = self.enc2(self.pool(e1))
		e3 = self.enc3(self.pool(e2))
		e4 = self.enc4(self.pool(e3))

		s3 = e3.shape[2:]
		d3 = self.d3_fuse(torch.cat([
			self.e1_d3(F.max_pool2d(e1, 4, 4)),
			self.e2_d3(F.max_pool2d(e2, 2, 2)),
			self.e3_d3(e3),
			self.e4_d3(self._resize(e4, s3)),
		], dim=1))

		s2 = e2.shape[2:]
		d2 = self.d2_fuse(torch.cat([
			self.e1_d2(F.max_pool2d(e1, 2, 2)),
			self.e2_d2(e2),
			self.e3_d2(self._resize(e3, s2)),
			self.e4_d2(self._resize(e4, s2)),
			self.d3_d2(self._resize(d3, s2)),
		], dim=1))

		s1 = e1.shape[2:]
		d1 = self.d1_fuse(torch.cat([
			self.e1_d1(e1),
			self.e2_d1(self._resize(e2, s1)),
			self.e3_d1(self._resize(e3, s1)),
			self.e4_d1(self._resize(e4, s1)),
			self.d3_d1(self._resize(d3, s1)),
			self.d2_d1(self._resize(d2, s1)),
		], dim=1))

		fused = self.refine(d1)
		return self.head(fused)

class Decoder(nn.Module):
	"""
	双分支解码器：
	- Locator: UNet++ 分割定位
	- Watermark Decoder: UNet3+ 骨干 + Conv mapping
	"""

	def __init__(self, H, W, message_length, base_ch=32):
		super().__init__()
		self.locator = LocatorUNetPP(in_ch=3, base_ch=base_ch)
		self.watermark_decoder = WatermarkDecoderUNet3Plus(
			message_length=message_length,
			in_ch=3,
			base_ch=base_ch,
			agg_ch=base_ch,
		)

	def forward_locator(self, noised_image):
		return self.locator(noised_image)

	def forward_decode(self, cropped_image):
		return self.watermark_decoder(cropped_image)

	def forward(self, noised_image, cropped_image=None):
		if cropped_image is None:
			cropped_image = noised_image
		return self.forward_decode(cropped_image)

class Decoder_Diffusion(nn.Module):
	"""
	保留接口，复用相同双分支结构。
	"""

	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
		super().__init__()
		self.decoder = Decoder(H, W, message_length)

	def forward(self, noised_image, cropped_image=None):
		return self.decoder(noised_image, cropped_image)
