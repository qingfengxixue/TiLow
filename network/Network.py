import kornia.losses
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .noise_layers.crop import Cropout
from .Encoder_MP_Decoder import *
from .Discriminator import Discriminator
from .TemplateWatermark import TemplateWatermark

class Network:

    def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion=False,
                 decoder_weight=1.0, grad_clip_norm=0.0, alpha=0.15,
                 gaussian_aug=False, gaussian_aug_prob=0.5, gaussian_sigma_min=0.0, gaussian_sigma_max=0.05,
                 use_cropout_attack=False, cropout_height_ratio=0.5, cropout_width_ratio=0.5,
                 locator_pretrain=False, single_tile_embed=False, random_tile_position=True,
                 locator_feature_supervision=False, locator_feat_weight=1.0,
                 locator_crop_size=128, locator_threshold=0.5, inputs_preprocessed=False,
                 use_second_noise=True):
        self.device = device
        self.message_length = message_length
        self.H = H
        self.W = W

        if not with_diffusion:
            self.encoder_decoder = EncoderDecoder(H, W, message_length, noise_layers).to(device)
        else:
            self.encoder_decoder = EncoderDecoder_Diffusion(H, W, message_length, noise_layers).to(device)

        self.discriminator = Discriminator().to(device)

        self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)
        self.discriminator = torch.nn.DataParallel(self.discriminator)

        self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
        self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

        print(lr)

        self.opt_locator = torch.optim.Adam(self.encoder_decoder.module.decoder.locator.parameters(), lr=lr)
        decoder_params = list(self.encoder_decoder.module.decoder.watermark_decoder.parameters())
        self.opt_decoder = torch.optim.Adam(filter(lambda p: p.requires_grad, decoder_params), lr=lr)
        self.opt_encoder_decoder = self.opt_decoder
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)

        self.discriminator_weight = 0.0001
        self.decoder_weight = decoder_weight
        self.grad_clip_norm = grad_clip_norm

        self.tile_size = 64
        self.crop_size = locator_crop_size
        self.locator_threshold = locator_threshold
        self.bit_loss_weight = decoder_weight
        self.locator_pretrain = locator_pretrain
        self.single_tile_embed = single_tile_embed
        self.random_tile_position = random_tile_position
        self.locator_feature_supervision = locator_feature_supervision
        self.locator_feat_weight = locator_feat_weight
        self.inputs_preprocessed = inputs_preprocessed
        self.use_second_noise = bool(use_second_noise)

        self.template_watermark = TemplateWatermark(
            d=8,
            alpha=alpha,
            tile_size=64,
            unit_size=8,
            single_tile=self.single_tile_embed,
            random_position=self.random_tile_position,
        )

        self.gaussian_aug = gaussian_aug
        self.gaussian_aug_prob = gaussian_aug_prob
        self.gaussian_sigma_min = gaussian_sigma_min
        self.gaussian_sigma_max = gaussian_sigma_max

        self.use_cropout_attack = use_cropout_attack
        self.cropout = Cropout(cropout_height_ratio, cropout_width_ratio) if use_cropout_attack else None

        self.locator_frozen = False
        self.current_stage = "locator" if locator_pretrain else "decoder"
        self.set_train_stage(self.current_stage)

    def _random_crop_triplet(self, cover_images: torch.Tensor, templated_images: torch.Tensor, noised_images: torch.Tensor):

        b, _, h_all, w_all = templated_images.shape
        if h_all <= 0 or w_all <= 0:
            offsets = [(0, 0)] * b
            return cover_images, templated_images, noised_images, offsets

        cropped_cover = []
        cropped_templated = []
        cropped_noised = []
        offsets = []

        out_side = int(self.tile_size * 2)

        for i in range(b):
            cover_i = cover_images[i:i + 1]

            content_mask = (cover_i.abs().sum(dim=1, keepdim=False) > 0)[0]
            if content_mask.any():
                ys, xs = torch.where(content_mask)
                y_min = int(ys.min().item())
                y_max = int(ys.max().item())
                x_min = int(xs.min().item())
                x_max = int(xs.max().item())
            else:
                y_min, y_max, x_min, x_max = 0, h_all - 1, 0, w_all - 1

            valid_h = max(1, y_max - y_min + 1)
            valid_w = max(1, x_max - x_min + 1)
            crop_side = min(self.tile_size * 2, valid_h, valid_w)

            if crop_side <= 0:
                top = y_min
                left = x_min
                crop_side = min(self.tile_size * 2, h_all, w_all)
            else:
                max_top = max(y_min, y_max - crop_side + 1)
                max_left = max(x_min, x_max - crop_side + 1)
                top = torch.randint(y_min, max_top + 1, (1,), device=templated_images.device).item()
                left = torch.randint(x_min, max_left + 1, (1,), device=templated_images.device).item()

            cover_patch = cover_images[i:i + 1, :, top:top + crop_side, left:left + crop_side]
            templated_patch = templated_images[i:i + 1, :, top:top + crop_side, left:left + crop_side]
            noised_patch = noised_images[i:i + 1, :, top:top + crop_side, left:left + crop_side]

            if cover_patch.shape[2] != out_side or cover_patch.shape[3] != out_side:
                cover_patch = F.interpolate(cover_patch, size=(out_side, out_side), mode="bilinear", align_corners=False)
            if templated_patch.shape[2] != out_side or templated_patch.shape[3] != out_side:
                templated_patch = F.interpolate(templated_patch, size=(out_side, out_side), mode="bilinear", align_corners=False)
            if noised_patch.shape[2] != out_side or noised_patch.shape[3] != out_side:
                noised_patch = F.interpolate(noised_patch, size=(out_side, out_side), mode="bilinear", align_corners=False)

            cropped_cover.append(cover_patch)
            cropped_templated.append(templated_patch)
            cropped_noised.append(noised_patch)
            offsets.append((int(top), int(left)))

        return torch.cat(cropped_cover, dim=0), torch.cat(cropped_templated, dim=0), torch.cat(cropped_noised, dim=0), offsets

    def _set_requires_grad(self, module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def set_train_stage(self, stage: str):
        if stage not in ("locator", "decoder"):
            raise ValueError("stage must be 'locator' or 'decoder'")
        self.current_stage = stage

        if stage == "locator":
            self._set_requires_grad(self.encoder_decoder.module.decoder.locator, True)
            self._set_requires_grad(self.encoder_decoder.module.decoder.watermark_decoder, False)
        else:
            self._set_requires_grad(self.encoder_decoder.module.decoder.watermark_decoder, True)
            self._set_requires_grad(self.encoder_decoder.module.decoder.locator, False)
            self.locator_frozen = True

    def freeze_locator(self):
        self._set_requires_grad(self.encoder_decoder.module.decoder.locator, False)
        self.locator_frozen = True

    def set_template_alpha(self, alpha_value: float):
        self.template_watermark.alpha = float(alpha_value)

    def forward_with_template(self, images: torch.Tensor, messages: torch.Tensor, cover_images: torch.Tensor = None):
        images, messages = images.to(self.device), messages.to(self.device)
        if cover_images is not None:
            cover_images = cover_images.to(self.device)

        if self.inputs_preprocessed:
            cover = cover_images if cover_images is not None else images
            offsets = [(0, 0)] * images.shape[0]
            encoded_images = images
            noised_images = images
            return cover, images, None, None, offsets, encoded_images, noised_images

        s, m_s, m, offsets = self.template_watermark(images, messages)

        noised_full = self.encoder_decoder.module.noise([s, images])
        images_cropped, s_cropped, noised_cropped, crop_offsets = self._random_crop_triplet(images, s, noised_full)

        if self.single_tile_embed:
            rel_offsets = []
            for i, (tile_top, tile_left) in enumerate(offsets):
                crop_top, crop_left = crop_offsets[i]
                rel_offsets.append((int(tile_top - crop_top), int(tile_left - crop_left)))
            offsets_embed = rel_offsets
        else:

            rel_offsets = []
            tile = int(self.tile_size)
            side = int(s_cropped.shape[2]) if s_cropped.shape[2] > 0 else tile * 2
            for crop_top, crop_left in crop_offsets:
                first_tile_top = ((int(crop_top) + tile - 1) // tile) * tile
                first_tile_left = ((int(crop_left) + tile - 1) // tile) * tile
                last_tile_top = ((int(crop_top) + side - tile) // tile) * tile
                last_tile_left = ((int(crop_left) + side - tile) // tile) * tile

                if first_tile_top <= last_tile_top:
                    num_top = ((last_tile_top - first_tile_top) // tile) + 1
                    tile_top = first_tile_top + int(torch.randint(0, num_top, (1,), device=images.device).item()) * tile
                else:
                    tile_top = int(crop_top)

                if first_tile_left <= last_tile_left:
                    num_left = ((last_tile_left - first_tile_left) // tile) + 1
                    tile_left = first_tile_left + int(torch.randint(0, num_left, (1,), device=images.device).item()) * tile
                else:
                    tile_left = int(crop_left)

                rel_offsets.append((int(tile_top - crop_top), int(tile_left - crop_left)))

            offsets_embed = rel_offsets

        encoded_images = s_cropped
        noised_images = noised_cropped

        return images_cropped, s_cropped, m_s, m, offsets_embed, encoded_images, noised_images

    def _dice_loss(self, logits, targets, smooth=1e-6):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def _build_locator_target_full(self, offsets, h, w, dtype=torch.float32):

        locator_map = self.template_watermark.build_locator_map(
            h=h,
            w=w,
            offsets=offsets,
            device=self.device,
            dtype=dtype,
        )
        return (locator_map > 0).float()

    def _crop_from_locator(self, images: torch.Tensor, locator_logits: torch.Tensor, offsets_embed=None):
        b, c, h, w = images.shape
        crops = []
        boxes = []

        for i in range(b):
            prob = torch.sigmoid(locator_logits[i, 0])
            mask = prob > self.locator_threshold

            if mask.any():
                ys, xs = torch.where(mask)
                y1, y2 = ys.min().item(), ys.max().item()
                x1, x2 = xs.min().item(), xs.max().item()
            elif offsets_embed is not None:
                top, left = offsets_embed[i]
                y1, y2 = int(top), int(top + self.tile_size - 1)
                x1, x2 = int(left), int(left + self.tile_size - 1)
            else:
                y1, y2 = 0, h - 1
                x1, x2 = 0, w - 1

            hh = y2 - y1 + 1
            ww = x2 - x1 + 1
            side = max(hh, ww, self.tile_size)
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2

            top = max(0, cy - side // 2)
            left = max(0, cx - side // 2)
            bottom = min(h, top + side)
            right = min(w, left + side)

            top = max(0, bottom - side)
            left = max(0, right - side)

            patch = images[i:i + 1, :, top:bottom, left:right]
            patch = F.interpolate(patch, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
            crops.append(patch)
            boxes.append((top, left, bottom, right))

        return torch.cat(crops, dim=0), boxes

    def _decoder_error_rate(self, messages: torch.Tensor, logits: torch.Tensor):
        preds = (torch.sigmoid(logits) > 0.5).float()
        return (preds != messages).float().mean().item()

    def train_locator(self, images: torch.Tensor, messages: torch.Tensor, cover_images: torch.Tensor = None,
                      locator_targets: torch.Tensor = None):
        self.set_train_stage("locator")
        self.encoder_decoder.train()

        with torch.enable_grad():
            images, messages = images.to(self.device), messages.to(self.device)
            images_proc, _, _, _, offsets_embed, encoded_images, noised_images = self.forward_with_template(images, messages, cover_images)

            s_sub = noised_images
            if locator_targets is not None:
                locator_target = locator_targets.to(self.device).float()
            else:
                locator_target = self._build_locator_target_full(offsets_embed, s_sub.shape[2], s_sub.shape[3], s_sub.dtype)
            locator_logit = self.encoder_decoder.module.decoder.forward_locator(s_sub)

            loc_loss = self._dice_loss(locator_logit, locator_target)

            self.opt_locator.zero_grad()
            loc_loss.backward()
            if self.grad_clip_norm and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.encoder_decoder.module.decoder.locator.parameters(), self.grad_clip_norm)
            self.opt_locator.step()

            psnr = -kornia.losses.psnr_loss(encoded_images.detach(), images_proc, 2)
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images_proc, window_size=5, reduction="mean")
            dice_score = (1.0 - loc_loss).detach()

        return {
            "locator_loss": loc_loss,
            "locator_dice": dice_score,
            "g_loss": loc_loss,
            "psnr": psnr,
            "ssim": ssim,
            "error_rate": 1.0 - dice_score,
            "g_loss_on_encoder": torch.tensor(0.0, device=self.device),
            "g_loss_on_decoder": loc_loss,
            "g_loss_on_discriminator": torch.tensor(0.0, device=self.device),
            "d_cover_loss": torch.tensor(0.0, device=self.device),
            "d_encoded_loss": torch.tensor(0.0, device=self.device),
        }

    def train_decoder(self, images: torch.Tensor, messages: torch.Tensor, cover_images: torch.Tensor = None,
                      locator_targets: torch.Tensor = None):
        self.set_train_stage("decoder")
        self.encoder_decoder.train()

        with torch.enable_grad():
            images, messages = images.to(self.device), messages.to(self.device)
            images_proc, _, _, _, offsets_embed, encoded_images, noised_images = self.forward_with_template(images, messages, cover_images)

            s_sub = noised_images
            if locator_targets is not None:
                locator_target = locator_targets.to(self.device).float()
            else:
                locator_target = self._build_locator_target_full(offsets_embed, s_sub.shape[2], s_sub.shape[3], s_sub.dtype)

            with torch.no_grad():
                locator_logit = self.encoder_decoder.module.decoder.forward_locator(s_sub)
            s_pred, _ = self._crop_from_locator(s_sub, locator_logit, offsets_embed)

            if self.use_cropout_attack and self.cropout is not None:
                s_pred = self.cropout([s_pred, s_pred])

            if self.gaussian_aug and np.random.rand() < self.gaussian_aug_prob:
                sigma = np.random.uniform(self.gaussian_sigma_min, self.gaussian_sigma_max)
                s_pred = (s_pred + torch.randn_like(s_pred) * sigma).clamp(-1, 1)

            if self.use_second_noise:
                s_pred = self.encoder_decoder.module.noise([s_pred, s_pred])
            wpred_logit = self.encoder_decoder.module.decoder.forward_decode(s_pred)
            dec_loss = self.criterion_BCE(wpred_logit, messages)
            enc_loss = torch.zeros((), device=self.device)
            g_loss = dec_loss

            self.opt_decoder.zero_grad()
            g_loss.backward()
            if self.grad_clip_norm and self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()),
                    self.grad_clip_norm,
                )
            self.opt_decoder.step()

            psnr = -kornia.losses.psnr_loss(encoded_images.detach(), images_proc, 2)
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images_proc, window_size=5, reduction="mean")
            error_rate = self._decoder_error_rate(messages, wpred_logit)
            locator_dice = 1.0 - self._dice_loss(locator_logit.detach(), locator_target)

        return {
            "error_rate": error_rate,
            "decoder_bce": dec_loss,
            "locator_dice": locator_dice,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss": g_loss,
            "g_loss_on_discriminator": torch.tensor(0.0, device=self.device),
            "g_loss_on_encoder": enc_loss,
            "g_loss_on_decoder": dec_loss,
            "d_cover_loss": torch.tensor(0.0, device=self.device),
            "d_encoded_loss": torch.tensor(0.0, device=self.device),
        }

    def train(self, images: torch.Tensor, messages: torch.Tensor, cover_images: torch.Tensor = None,
              locator_targets: torch.Tensor = None):
        if self.current_stage == "locator":
            return self.train_locator(images, messages, cover_images, locator_targets)
        return self.train_decoder(images, messages, cover_images, locator_targets)

    def validation_locator(self, images: torch.Tensor, messages: torch.Tensor, cover_images: torch.Tensor = None,
                           locator_targets: torch.Tensor = None):
        self.encoder_decoder.eval()

        with torch.no_grad():
            images, messages = images.to(self.device), messages.to(self.device)
            images_proc, _, _, _, offsets_embed, encoded_images, noised_images = self.forward_with_template(images, messages, cover_images)

            s_sub = noised_images
            if locator_targets is not None:
                locator_target = locator_targets.to(self.device).float()
            else:
                locator_target = self._build_locator_target_full(offsets_embed, s_sub.shape[2], s_sub.shape[3], s_sub.dtype)
            locator_logit = self.encoder_decoder.module.decoder.forward_locator(s_sub)

            loc_loss = self._dice_loss(locator_logit, locator_target)
            dice_score = 1.0 - loc_loss
            psnr = -kornia.losses.psnr_loss(encoded_images.detach(), images_proc, 2)
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images_proc, window_size=5, reduction="mean")

        result = {
            "locator_loss": loc_loss,
            "locator_dice": dice_score,
            "g_loss": loc_loss,
            "psnr": psnr,
            "ssim": ssim,
            "error_rate": 1.0 - dice_score,
            "g_loss_on_discriminator": torch.tensor(0.0, device=self.device),
            "g_loss_on_encoder": torch.tensor(0.0, device=self.device),
            "g_loss_on_decoder": loc_loss,
            "d_cover_loss": torch.tensor(0.0, device=self.device),
            "d_encoded_loss": torch.tensor(0.0, device=self.device),
        }

        return result, (images_proc, s_sub, encoded_images, noised_images, messages, None, list(range(images.shape[0])), locator_logit, locator_target)

    def validation_decoder(self, images: torch.Tensor, messages: torch.Tensor, cover_images: torch.Tensor = None,
                           locator_targets: torch.Tensor = None):
        self.encoder_decoder.eval()

        with torch.no_grad():
            images, messages = images.to(self.device), messages.to(self.device)
            images_proc, _, _, _, offsets_embed, encoded_images, noised_images = self.forward_with_template(images, messages, cover_images)

            s_sub = noised_images
            if locator_targets is not None:
                locator_target = locator_targets.to(self.device).float()
            else:
                locator_target = self._build_locator_target_full(offsets_embed, s_sub.shape[2], s_sub.shape[3], s_sub.dtype)
            locator_logit = self.encoder_decoder.module.decoder.forward_locator(s_sub)
            s_pred, _ = self._crop_from_locator(s_sub, locator_logit, offsets_embed)
            s_pred = self.encoder_decoder.module.noise([s_pred, s_pred])

            wpred_logit = self.encoder_decoder.module.decoder.forward_decode(s_pred)
            dec_loss = self.criterion_BCE(wpred_logit, messages)
            enc_loss = torch.zeros((), device=self.device)
            g_loss = dec_loss

            psnr = -kornia.losses.psnr_loss(encoded_images.detach(), images_proc, 2)
            ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images_proc, window_size=5, reduction="mean")
            error_rate = self._decoder_error_rate(messages, wpred_logit)
            locator_dice = 1.0 - self._dice_loss(locator_logit, locator_target)

        result = {
            "error_rate": error_rate,
            "decoder_bce": dec_loss,
            "locator_dice": locator_dice,
            "psnr": psnr,
            "ssim": ssim,
            "g_loss": g_loss,
            "g_loss_on_discriminator": torch.tensor(0.0, device=self.device),
            "g_loss_on_encoder": enc_loss,
            "g_loss_on_decoder": dec_loss,
            "d_cover_loss": torch.tensor(0.0, device=self.device),
            "d_encoded_loss": torch.tensor(0.0, device=self.device),
        }

        return result, (images_proc, s_sub, encoded_images, noised_images, messages, wpred_logit, list(range(images.shape[0])), locator_logit, locator_target)

    def validation(self, images: torch.Tensor, messages: torch.Tensor, cover_images: torch.Tensor = None,
                   locator_targets: torch.Tensor = None):
        if self.current_stage == "locator":
            return self.validation_locator(images, messages, cover_images, locator_targets)
        return self.validation_decoder(images, messages, cover_images, locator_targets)

    def decoded_message_error_rate_batch(self, messages, decoded_messages, batch_indices=None):
        if decoded_messages is None:
            return 1.0

        if decoded_messages.dim() == 2:
            return self._decoder_error_rate(messages, decoded_messages)

        if decoded_messages.dim() == 4:
            probs = torch.sigmoid(decoded_messages)
            preds = (probs > 0.5).float()
            if preds.shape[1] != messages.shape[1]:
                return 1.0
            preds = preds.mean(dim=(2, 3))
            preds = (preds > 0.5).float()
            return (preds != messages).float().mean().item()

        return 1.0

    def save_model(self, path_encoder_decoder: str, path_discriminator: str):
        torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
        torch.save(self.discriminator.module.state_dict(), path_discriminator)

    def load_model(self, path_encoder_decoder: str, path_discriminator: str):
        self.load_model_ed(path_encoder_decoder)
        self.load_model_dis(path_discriminator)

    def load_model_ed(self, path_encoder_decoder: str):
        self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder, map_location=self.device))

    def load_model_dis(self, path_discriminator: str):
        self.discriminator.module.load_state_dict(torch.load(path_discriminator, map_location=self.device))
