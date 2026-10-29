import os
import sys
import numpy as np
import torch
import kornia.losses
from PIL import Image
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import hid_utils
from utils import *
from network.Network import *
from network.TemplateWatermark import TemplateWatermark
from utils.load_train_setting import *

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEP_LOG_INTERVAL = 20

def save_locator_heatmap(locator_logit, locator_target, epoch, step, folder):
    if locator_logit is None or locator_target is None:
        return

    loc_pred = torch.sigmoid(locator_logit[:1, 0]).detach().cpu().numpy()
    loc_tgt = locator_target[:1, 0].detach().cpu().numpy()
    loc_pred = np.squeeze(loc_pred)
    loc_tgt = np.squeeze(loc_tgt)
    loc_pred = (loc_pred - loc_pred.min()) / (loc_pred.max() - loc_pred.min() + 1e-6)
    loc_tgt = (loc_tgt - loc_tgt.min()) / (loc_tgt.max() - loc_tgt.min() + 1e-6)
    img_pred = (loc_pred * 255).astype(np.uint8)
    img_tgt = (loc_tgt * 255).astype(np.uint8)
    stack = np.concatenate([img_pred, img_tgt], axis=1)
    os.makedirs(folder, exist_ok=True)
    Image.fromarray(stack).convert("L").save(os.path.join(folder, f"loc_{epoch}_{step}.png"))

def update_running(running, result):
    for key, value in result.items():
        if hasattr(value, "detach"):
            value = value.detach().item()
        else:
            value = float(value)
        running[key] = running.get(key, 0.0) + value
        if key == "error_rate":
            running["error_rate_sq"] = running.get("error_rate_sq", 0.0) + value * value

def avg_dict(running, num):
    if num <= 0:
        return {}
    return {k: (v / num) for k, v in running.items()}

def format_epoch_content(epoch, phase, stage, epoch_alpha, running, num):
    content = f"Epoch {epoch} stage={stage} phase={phase} alpha={epoch_alpha:.6f}\n"
    for k, v in avg_dict(running, num).items():
        content += f"{k}={v},"
    content += "\n"
    return content

def linear_alpha(epoch, start_epoch, end_epoch, start_alpha, end_alpha):
    if end_epoch <= start_epoch:
        return float(end_alpha)
    if epoch <= start_epoch:
        return float(start_alpha)
    if epoch >= end_epoch:
        return float(end_alpha)
    ratio = float(epoch - start_epoch) / float(end_epoch - start_epoch)
    return float(start_alpha + (end_alpha - start_alpha) * ratio)

def locator_alpha_schedule(stage_epoch, decay_start_epoch):

    start_value = float(locator_alpha_phase1_start)
    drop1_step = float(locator_alpha_drop1_step)
    drop2_step = float(locator_alpha_drop2_step)
    drop2_start_value = float(locator_alpha_drop2_start_value)

    start_epoch = max(1, int(decay_start_epoch))
    drop1_end_epoch = start_epoch + 3
    drop2_start_epoch = drop1_end_epoch + 1

    if stage_epoch < start_epoch:
        return start_value

    if stage_epoch <= drop1_end_epoch:
        return start_value - drop1_step * float(stage_epoch - start_epoch + 1)

    return drop2_start_value - drop2_step * float(stage_epoch - drop2_start_epoch)

def _sample_vote_crops(h, w, crop_size, n_crops, tile_size=None):
    max_top = h - crop_size
    max_left = w - crop_size
    if max_top < 0 or max_left < 0:
        return []

    n_crops = max(1, int(n_crops))
    if n_crops % 2 == 0:
        n_crops += 1

    top_min, top_max = 0, max_top
    left_min, left_max = 0, max_left

    coords = []
    used = set()
    attempts = 0
    max_attempts = max(50, n_crops * 20)

    while len(coords) < n_crops and attempts < max_attempts:
        attempts += 1
        top = int(torch.randint(top_min, top_max + 1, (1,)).item())
        left = int(torch.randint(left_min, left_max + 1, (1,)).item())

        if tile_size is not None and tile_size > 0:
            if max_top > 0 and top % tile_size == 0:
                top = min(max_top, top + 1)
                if top % tile_size == 0:
                    top = max(0, top - 1)
            if max_left > 0 and left % tile_size == 0:
                left = min(max_left, left + 1)
                if left % tile_size == 0:
                    left = max(0, left - 1)

        key = (top, left)
        if key in used:
            continue
        used.add(key)
        coords.append(key)

    if not coords:
        coords.append((0, 0))
    return coords

def _adaptive_vote_count(h, w, crop_size):
    w_space = max(0, int(w - crop_size))
    h_space = max(0, int(h - crop_size))
    d_space = min(w_space, h_space)

    if d_space >= 2 * crop_size:
        k = 4
    elif d_space >= crop_size:
        k = 3
    elif d_space >= crop_size // 2:
        k = 2
    else:
        k = 1

    k = max(1, min(k, 5))
    n_crops = 2 * k + 1
    return w_space, h_space, d_space, k, n_crops

def run_epoch_test_stage(network, test_loader, epoch, epoch_alpha):
    if test_loader is None:
        return {}

    template = TemplateWatermark(
        d=8,
        alpha=epoch_alpha,
        tile_size=64,
        unit_size=8,
        single_tile=single_tile_embed,
        random_position=random_tile_position,
    )

    running = {}
    steps = 0
    test_steps = int(test_stage_steps)
    a = int(template.tile_size)
    crop_size = int(2 * a)

    network.encoder_decoder.eval()
    with torch.no_grad():
        for i, image in enumerate(test_loader):
            if test_steps > 0 and i >= test_steps:
                break

            image = image.to(device)
            if image.dim() == 3:
                image = image.unsqueeze(0)

            if fixed_message:
                msg = torch.Tensor(np.random.RandomState(fixed_message_seed).choice([0, 1], (1, message_length))).to(device)
            else:
                msg = torch.Tensor(np.random.choice([0, 1], (1, message_length))).to(device)

            templated, _, _, _ = template(image, msg)

            _, _, h_t, w_t = templated.shape
            if h_t < crop_size or w_t < crop_size:
                continue

            if use_vote_mechanism:
                w_space, h_space, d_space, vote_k, vote_n = _adaptive_vote_count(h_t, w_t, crop_size)
                crop_coords = _sample_vote_crops(h_t, w_t, crop_size, vote_n, tile_size=a)
            else:
                w_space = max(0, int(w_t - crop_size))
                h_space = max(0, int(h_t - crop_size))
                d_space = min(w_space, h_space)
                vote_k = 1
                vote_n = 1
                crop_coords = _sample_vote_crops(h_t, w_t, crop_size, 1, tile_size=a)

            crop_probs = []
            psnr_list = []
            ssim_list = []
            for top, left in crop_coords:
                patch_templated = templated[:, :, top:top + crop_size, left:left + crop_size]
                patch_cover = image[:, :, top:top + crop_size, left:left + crop_size]
                if patch_templated.shape[2] != crop_size or patch_templated.shape[3] != crop_size:
                    continue
                if patch_cover.shape[2] != crop_size or patch_cover.shape[3] != crop_size:
                    continue

                noised = network.encoder_decoder.module.noise([patch_templated, patch_cover])
                psnr = -kornia.losses.psnr_loss(patch_templated.detach(), patch_cover, 2).cpu().item()
                ssim = 1 - 2 * kornia.losses.ssim_loss(
                    patch_templated.detach(), patch_cover, window_size=5, reduction="mean"
                ).cpu().item()
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                locator_logit = network.encoder_decoder.module.decoder.forward_locator(noised)
                corrected, _ = network._crop_from_locator(noised, locator_logit, offsets_embed=None)
                pred_logit = network.encoder_decoder.module.decoder.forward_decode(corrected)
                pred_prob = torch.sigmoid(pred_logit).squeeze(0)
                crop_probs.append(pred_prob)

            if len(crop_probs) == 0:
                continue

            bits_single = (crop_probs[0] >= 0.5).float()
            error_single = (bits_single != msg.squeeze(0)).float().mean().item()
            if use_vote_mechanism:
                prob_stack = torch.stack(crop_probs, dim=0)
                prob_mean = prob_stack.mean(dim=0)
                bits_prob_vote = (prob_mean >= 0.5).float()
                error_prob_vote = (bits_prob_vote != msg.squeeze(0)).float().mean().item()
            else:
                error_prob_vote = error_single
            avg_psnr = float(np.mean(psnr_list)) if len(psnr_list) > 0 else 0.0
            avg_ssim = float(np.mean(ssim_list)) if len(ssim_list) > 0 else 0.0

            step_result = {
                "error_rate": error_prob_vote,
                "error_rate_single": error_single,
                "error_rate_vote": error_prob_vote,
                "psnr": avg_psnr,
                "ssim": avg_ssim,
                "vote_crops": float(len(crop_probs)),
                "vote_k": float(vote_k),
                "vote_d_space": float(d_space),
                "vote_w_space": float(w_space),
                "vote_h_space": float(h_space),
            }

            if i % STEP_LOG_INTERVAL == 0:
                print('----------------------------------------------------')
                print(f'Epoch: {epoch} Stage: test Alpha: {epoch_alpha:.6f} Step: {i}')
                for key, value in step_result.items():
                    print(f"{key}={value}")

            update_running(running, step_result)
            steps += 1

    content = format_epoch_content(epoch, "test", "test", epoch_alpha, running, steps)
    print('-------------------------------------------------------')
    print("test")
    print(content)

    with open(result_folder + "/test_log.txt", "a") as file:
        file.write(content)

    if steps > 0:
        hid_utils.write_losses(os.path.join(result_folder, 'test.csv'), running, epoch, steps, 0)

    return avg_dict(running, steps)

def build_template_collate_fn(message_len, alpha_val, single_tile, random_tile, fixed_msg, fixed_seed, tile_size=64):

    template = TemplateWatermark(
        d=8,
        alpha=alpha_val,
        tile_size=tile_size,
        unit_size=8,
        single_tile=single_tile,
        random_position=random_tile,
    )

    target_builder = TemplateWatermark(
        d=8,
        alpha=alpha_val,
        tile_size=tile_size,
        unit_size=8,
        single_tile=True,
        random_position=False,
    )
    crop_side = tile_size * 2
    fixed_message_tensor = None
    if fixed_msg:
        fixed_message_tensor = torch.Tensor(np.random.RandomState(fixed_seed).choice([0, 1], (message_len,)))

    def collate_fn(batch):
        cover_images = []
        processed_images = []
        processed_messages = []
        locator_targets = []
        for image in batch:
            if fixed_message_tensor is not None:
                message = fixed_message_tensor.clone()
            else:
                message = torch.Tensor(np.random.choice([0, 1], (message_len,)))

            image_4d = image.unsqueeze(0)
            message_4d = message.unsqueeze(0)
            templated, _, _, offsets = template(image_4d, message_4d)

            _, _, h, w = templated.shape
            side = min(crop_side, h, w)
            max_top = max(0, h - side)
            max_left = max(0, w - side)

            if template.single_tile:
                tile_top, tile_left = offsets[0]

                top_low = max(0, tile_top - (side - tile_size))
                top_high = min(tile_top, max_top)
                left_low = max(0, tile_left - (side - tile_size))
                left_high = min(tile_left, max_left)

                if top_low > top_high:
                    top = max(0, min(tile_top, max_top))
                else:
                    top = torch.randint(top_low, top_high + 1, (1,)).item()
                if left_low > left_high:
                    left = max(0, min(tile_left, max_left))
                else:
                    left = torch.randint(left_low, left_high + 1, (1,)).item()

                rel_top = int(tile_top - top)
                rel_left = int(tile_left - left)
            else:

                if max_top <= 0:
                    top = 0
                else:
                    top = torch.randint(0, max_top + 1, (1,)).item()
                    if top % tile_size == 0:
                        top = min(max_top, top + 1)
                        if top % tile_size == 0:
                            top = max(0, top - 1)

                if max_left <= 0:
                    left = 0
                else:
                    left = torch.randint(0, max_left + 1, (1,)).item()
                    if left % tile_size == 0:
                        left = min(max_left, left + 1)
                        if left % tile_size == 0:
                            left = max(0, left - 1)

                first_tile_top = ((top + tile_size - 1) // tile_size) * tile_size
                first_tile_left = ((left + tile_size - 1) // tile_size) * tile_size
                last_tile_top = ((top + side - tile_size) // tile_size) * tile_size
                last_tile_left = ((left + side - tile_size) // tile_size) * tile_size

                if first_tile_top <= last_tile_top:
                    num_top = ((last_tile_top - first_tile_top) // tile_size) + 1
                    tile_top = first_tile_top + int(torch.randint(0, num_top, (1,)).item()) * tile_size
                else:
                    tile_top = max(0, min(top, h - tile_size))

                if first_tile_left <= last_tile_left:
                    num_left = ((last_tile_left - first_tile_left) // tile_size) + 1
                    tile_left = first_tile_left + int(torch.randint(0, num_left, (1,)).item()) * tile_size
                else:
                    tile_left = max(0, min(left, w - tile_size))

                rel_top = int(tile_top - top)
                rel_left = int(tile_left - left)

            cover = image_4d[:, :, top:top + side, left:left + side]
            cropped = templated[:, :, top:top + side, left:left + side]
            loc_target = target_builder.build_locator_map(
                h=side,
                w=side,
                offsets=[(rel_top, rel_left)],
                device=image.device,
                dtype=image.dtype,
            )
            loc_target = (loc_target > 0).float()

            if side != crop_side:
                cover = torch.nn.functional.interpolate(cover, size=(crop_side, crop_side), mode="bilinear", align_corners=False)
                cropped = torch.nn.functional.interpolate(cropped, size=(crop_side, crop_side), mode="bilinear", align_corners=False)
                loc_target = torch.nn.functional.interpolate(loc_target, size=(crop_side, crop_side), mode="nearest")

            cover_images.append(cover.squeeze(0))
            processed_images.append(cropped.squeeze(0))
            processed_messages.append(message)
            locator_targets.append(loc_target.squeeze(0))

        return (
            torch.stack(cover_images, dim=0),
            torch.stack(processed_images, dim=0),
            torch.stack(processed_messages, dim=0),
            torch.stack(locator_targets, dim=0),
        )

    def set_alpha(new_alpha):
        template.alpha = float(new_alpha)
        target_builder.alpha = float(new_alpha)

    collate_fn.set_alpha = set_alpha
    collate_fn.get_alpha = lambda: float(template.alpha)

    return collate_fn

def build_pad_collate_fn(pad_value=0.0):

    def collate_fn(batch):
        if len(batch) == 0:
            return torch.empty(0)

        max_h = max(int(img.shape[1]) for img in batch)
        max_w = max(int(img.shape[2]) for img in batch)

        padded = []
        for img in batch:
            c, h, w = img.shape
            canvas = torch.full((c, max_h, max_w), float(pad_value), dtype=img.dtype)
            canvas[:, :h, :w] = img
            padded.append(canvas)

        return torch.stack(padded, dim=0)

    return collate_fn

network = Network(
    H,
    W,
    message_length,
    noise_layers,
    device,
    batch_size,
    lr,
    with_diffusion=with_diffusion,
    decoder_weight=decoder_weight,
    grad_clip_norm=grad_clip_norm,
    alpha=alpha,
    gaussian_aug=gaussian_aug,
    gaussian_aug_prob=gaussian_aug_prob,
    gaussian_sigma_min=gaussian_sigma_min,
    gaussian_sigma_max=gaussian_sigma_max,
    use_cropout_attack=use_cropout_attack,
    cropout_height_ratio=cropout_height_ratio,
    cropout_width_ratio=cropout_width_ratio,
    locator_pretrain=locator_pretrain,
    single_tile_embed=single_tile_embed,
    random_tile_position=random_tile_position,
    locator_feature_supervision=locator_feature_supervision,
    locator_feat_weight=locator_feat_weight,
    locator_crop_size=locator_crop_size,
    locator_threshold=locator_threshold,
    inputs_preprocessed=preprocess_before_batch,
    use_second_noise=use_second_noise,
)

collate_fn = None
if preprocess_before_batch:
    collate_fn = build_template_collate_fn(
        message_len=message_length,
        alpha_val=alpha,
        single_tile=single_tile_embed,
        random_tile=random_tile_position,
        fixed_msg=fixed_message,
        fixed_seed=fixed_message_seed,
        tile_size=64,
    )
else:
    collate_fn = build_pad_collate_fn(pad_value=0.0)

train_dataset = Dataset(os.path.join(dataset_path, "train"), H, W)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              collate_fn=collate_fn)

val_dataset = Dataset(os.path.join(dataset_path, "val"), H, W)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=collate_fn)

test_dataloader = None
test_folder = os.path.join(dataset_path, "test")
if enable_epoch_test_stage and os.path.isdir(test_folder):
    test_dataset = Dataset(test_folder, H, W)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

if debug_test_only:
    print("debug_test_only=True, skip train/val and run test stage only.")
    run_epoch_test_stage(network, test_dataloader, train_continue_epoch if train_continue else 0, float(alpha))
    raise SystemExit(0)

if train_continue:
    EC_path = "results/" + train_continue_path + "/models/EC_" + str(train_continue_epoch) + ".pth"
    D_path = "results/" + train_continue_path + "/models/D_" + str(train_continue_epoch) + ".pth"
    network.load_model(EC_path, D_path)
elif decoder_only_init_path:
    network.load_model_ed(decoder_only_init_path)
    print("Loaded decoder-only init:", decoder_only_init_path)

if quick_train:
    max_steps = quick_steps
    total_epoch_budget = quick_epochs
else:
    max_steps = None
    total_epoch_budget = epoch_number

if two_stage_training:
    if quick_train:
        loc_epoch_budget = min(locator_stage_epochs, total_epoch_budget)
        dec_epoch_budget = max(total_epoch_budget - loc_epoch_budget, 0)
    else:
        loc_epoch_budget = locator_stage_epochs
        if decoder_stage_epochs > 0:
            dec_epoch_budget = decoder_stage_epochs
        else:
            dec_epoch_budget = max(total_epoch_budget - loc_epoch_budget, 0)
else:
    loc_epoch_budget = 0
    dec_epoch_budget = total_epoch_budget

current_stage = "locator" if two_stage_training and loc_epoch_budget > 0 else "decoder"
network.set_train_stage(current_stage)

print("\nStart training : \n")
print(f"two_stage_training={two_stage_training}, locator_epochs={loc_epoch_budget}, decoder_epochs={dec_epoch_budget}")

start = 1
locator_done = 0
decoder_done = 0
converge_count = 0
last_train_avg = {}
last_val_avg = {}
last_test_avg = {}

epoch_global = 0
last_epoch_alpha = float(alpha)
alpha_lock_active = False
alpha_locked_value = float(alpha)
adaptive_alpha_current = float(stage2_alpha_adaptive_start)
locator_decay_start_epoch = int(locator_alpha_drop1_start_epoch)
locator_prev_loss = None
locator_plateau_delta = float(locator_alpha_plateau_delta)
locator_plateau_monitor_epochs = int(locator_alpha_plateau_monitor_epochs)
locator_early_stop_alpha = float(locator_alpha_early_stop_alpha)
locator_early_stop_enable = bool(locator_alpha_early_stop_enable)
locator_force_stage2 = False
while True:
    if current_stage == "locator" and (locator_done >= loc_epoch_budget or locator_force_stage2):

        if locator_force_stage2:
            dec_epoch_budget = max(total_epoch_budget - locator_done, 0)
            print(f"locator_early_switch_adjust_decoder_epochs={dec_epoch_budget}")
        network.freeze_locator()
        current_stage = "decoder"
        network.set_train_stage("decoder")
        adaptive_alpha_current = float(stage2_alpha_adaptive_start)
        if locator_force_stage2:
            print("Locator stage finished early by plateau rule, switch to decoder stage.")
        else:
            print("Locator stage finished by epoch budget, switch to decoder stage.")
        locator_force_stage2 = False

    if current_stage == "decoder" and decoder_done >= dec_epoch_budget:
        break

    epoch_global += 1
    epoch = epoch_global + (train_continue_epoch if train_continue else 0)
    locator_stage_epoch = locator_done + 1

    epoch_alpha = float(alpha)
    if current_stage == "locator" and locator_alpha_decay:
        epoch_alpha = locator_alpha_schedule(locator_stage_epoch, locator_decay_start_epoch)

    if current_stage == "decoder" and stage2_alpha_adaptive:
        epoch_alpha = adaptive_alpha_current

    if current_stage == "decoder" and (not stage2_alpha_adaptive) and stage2_alpha_decay:
        p1_start = int(stage2_alpha_phase1_start_epoch)
        p1_end = int(stage2_alpha_phase1_end_epoch)
        p1_value = float(stage2_alpha_phase1_value)
        p2_end = int(stage2_alpha_phase2_end_epoch)
        p2_end_alpha = float(stage2_alpha_phase2_end)
        p3_end = int(stage2_alpha_phase3_end_epoch)
        p3_end_alpha = float(stage2_alpha_phase3_end)

        if epoch <= p1_end:
            epoch_alpha = p1_value
        elif epoch <= p2_end:
            epoch_alpha = linear_alpha(epoch, p1_end, p2_end, p1_value, p2_end_alpha)
        elif epoch <= p3_end:
            epoch_alpha = linear_alpha(epoch, p2_end, p3_end, p2_end_alpha, p3_end_alpha)
        else:
            epoch_alpha = p3_end_alpha

        if epoch < p1_start:
            epoch_alpha = p1_value

    if current_stage == "decoder" and (not stage2_alpha_adaptive) and stage2_alpha_hold_on_high_error and alpha_lock_active:
        epoch_alpha = alpha_locked_value

    if collate_fn is not None and hasattr(collate_fn, "set_alpha"):
        collate_fn.set_alpha(epoch_alpha)
    network.set_template_alpha(epoch_alpha)
    last_epoch_alpha = epoch_alpha
    print(f"epoch_alpha={epoch_alpha}, alpha_locked={alpha_lock_active}, alpha_adaptive={stage2_alpha_adaptive}")

    train_running = {}
    num = 0

    for i, images in enumerate(train_dataloader):
        if max_steps is not None and i >= max_steps:
            break

        cover_image = None
        locator_target = None
        if isinstance(images, (tuple, list)) and len(images) == 4:
            cover_image = images[0].to(device)
            image = images[1].to(device)
            message = images[2].to(device)
            locator_target = images[3].to(device)
        elif isinstance(images, (tuple, list)) and len(images) == 3:
            cover_image = images[0].to(device)
            image = images[1].to(device)
            message = images[2].to(device)
        else:
            image = images.to(device)
            if fixed_message:
                message = torch.Tensor(np.random.RandomState(fixed_message_seed).choice([0, 1], (1, message_length))).to(device)
                message = message.repeat(image.shape[0], 1)
            else:
                message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

        if current_stage == "locator":
            result = network.train_locator(image, message, cover_images=cover_image, locator_targets=locator_target)
        else:
            result = network.train_decoder(image, message, cover_images=cover_image, locator_targets=locator_target)

        if i % STEP_LOG_INTERVAL == 0:
            print('----------------------------------------------------')
            print(f'Epoch: {epoch} Stage: {current_stage} Alpha: {epoch_alpha:.6f} Step: {i}')
            for key in result:
                value = result[key].detach().item() if hasattr(result[key], "detach") else float(result[key])
                print(f"{key}={value}")

        update_running(train_running, result)
        num += 1

    train_content = format_epoch_content(epoch, "train", current_stage, epoch_alpha, train_running, num)
    print('-------------------------------------------------------')
    print("train")
    print(train_content)

    with open(result_folder + "/train_log.txt", "a") as file:
        file.write(train_content)

    hid_utils.write_losses(os.path.join(result_folder, 'train.csv'), train_running, epoch, num, start)
    last_train_avg = avg_dict(train_running, num)

    val_running = {}

    num = 0
    for i, images in enumerate(val_dataloader):
        if max_steps is not None and i >= max_steps:
            break

        cover_image = None
        locator_target = None
        if isinstance(images, (tuple, list)) and len(images) == 4:
            cover_image = images[0].to(device)
            image = images[1].to(device)
            message = images[2].to(device)
            locator_target = images[3].to(device)
        elif isinstance(images, (tuple, list)) and len(images) == 3:
            cover_image = images[0].to(device)
            image = images[1].to(device)
            message = images[2].to(device)
        else:
            image = images.to(device)
            if fixed_message:
                message = torch.Tensor(np.random.RandomState(fixed_message_seed).choice([0, 1], (1, message_length))).to(device)
                message = message.repeat(image.shape[0], 1)
            else:
                message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

        if current_stage == "locator":
            result, (images_v, s, encoded_images, noised_images, messages, decoded_messages, batch_indices, locator_logit, locator_target) = network.validation_locator(
                image, message, cover_images=cover_image, locator_targets=locator_target
            )
        else:
            result, (images_v, s, encoded_images, noised_images, messages, decoded_messages, batch_indices, locator_logit, locator_target) = network.validation_decoder(
                image, message, cover_images=cover_image, locator_targets=locator_target
            )

        if i % STEP_LOG_INTERVAL == 0:
            print('----------------------------------------------------')
            print(f'Epoch: {epoch} Stage: {current_stage} Alpha: {epoch_alpha:.6f} Step: {i}')
            for key in result:
                value = result[key].detach().item() if hasattr(result[key], "detach") else float(result[key])
                print(f"{key}={value}")

        update_running(val_running, result)
        num += 1

        if current_stage == "locator" and i < 2:
            save_locator_heatmap(locator_logit, locator_target, epoch, i, result_folder + "images/")

    val_content = format_epoch_content(epoch, "val", current_stage, epoch_alpha, val_running, num)
    print('-------------------------------------------------------')
    print("validation")
    print(val_content)

    with open(result_folder + "/val_log.txt", "a") as file:
        file.write(val_content)

    hid_utils.write_losses(os.path.join(result_folder, 'val.csv'), val_running, epoch, num, start)
    start = 0

    last_val_avg = avg_dict(val_running, num)

    if enable_epoch_test_stage:
        last_test_avg = run_epoch_test_stage(network, test_dataloader, epoch, epoch_alpha)
    else:
        last_test_avg = {}

    if current_stage == "locator":
        locator_done += 1
        train_locator_loss = float(last_train_avg.get("locator_loss", 1.0))
        if locator_prev_loss is not None:
            loss_delta = abs(train_locator_loss - locator_prev_loss)
            print(
                f"locator_loss_delta={loss_delta:.6f}, current_loss={train_locator_loss:.6f}, "
                f"prev_loss={locator_prev_loss:.6f}"
            )

            if locator_done <= locator_plateau_monitor_epochs and loss_delta < locator_plateau_delta:
                candidate_start = locator_done + 1
                if candidate_start < locator_decay_start_epoch:
                    locator_decay_start_epoch = candidate_start
                    print(
                        f"locator_alpha_early_decay=ON, reason=loss_delta({loss_delta:.6f})<"
                        f"threshold({locator_plateau_delta:.6f}), next_decay_start_epoch={locator_decay_start_epoch}"
                    )

            if locator_early_stop_enable and epoch_alpha <= locator_early_stop_alpha + 1e-8 and loss_delta < locator_plateau_delta:
                locator_force_stage2 = True
                loc_epoch_budget = locator_done
                print(
                    f"locator_stage_early_stop=ON, reason=alpha({epoch_alpha:.6f})<="
                    f"target({locator_early_stop_alpha:.6f}) and loss_delta({loss_delta:.6f})<"
                    f"threshold({locator_plateau_delta:.6f})"
                )
        locator_prev_loss = train_locator_loss

        val_dice = float(last_val_avg.get("locator_dice", 0.0))
        if val_dice >= locator_converge_threshold:
            converge_count += 1
        else:
            converge_count = 0

        print(f"locator_converge_status: val_dice={val_dice}, threshold={locator_converge_threshold}, hit={converge_count}/{locator_converge_patience}")
    else:
        decoder_done += 1
        if stage2_alpha_adaptive:
            test_error_rate_vote = float(last_test_avg.get("error_rate_vote", 1.0))
            threshold = float(stage2_alpha_adaptive_threshold)
            if test_error_rate_vote < threshold:
                switch_alpha = float(stage2_alpha_adaptive_switch)
                switch_alpha2 = float(stage2_alpha_adaptive_switch2)
                step_high = float(stage2_alpha_adaptive_step_high)
                step_low = float(stage2_alpha_adaptive_step_low)
                step_min = float(stage2_alpha_adaptive_step_min)
                min_alpha = float(stage2_alpha_adaptive_min)
                eps = 1e-8
                if adaptive_alpha_current > switch_alpha + eps:
                    next_alpha = adaptive_alpha_current - step_high
                    if next_alpha < switch_alpha:
                        next_alpha = switch_alpha
                elif adaptive_alpha_current > switch_alpha2 + eps:
                    next_alpha = adaptive_alpha_current - step_low
                    if next_alpha < switch_alpha2:
                        next_alpha = switch_alpha2
                else:
                    next_alpha = adaptive_alpha_current - step_min

                next_alpha = max(min_alpha, next_alpha)
                step = adaptive_alpha_current - next_alpha
                if next_alpha < adaptive_alpha_current:
                    print(
                        f"alpha_adaptive=DOWN, reason=test_error_rate_vote({test_error_rate_vote:.6f})<threshold({threshold:.6f}), "
                        f"alpha:{adaptive_alpha_current:.6f}->{next_alpha:.6f}, step={step:.6f}"
                    )
                adaptive_alpha_current = next_alpha
            else:
                print(
                    f"alpha_adaptive=HOLD, reason=test_error_rate_vote({test_error_rate_vote:.6f})>threshold({threshold:.6f}), "
                    f"alpha={adaptive_alpha_current:.6f}"
                )
        elif stage2_alpha_hold_on_high_error:
            val_error_rate = float(last_val_avg.get("error_rate", 1.0))
            threshold = float(stage2_alpha_errorrate_threshold)
            if val_error_rate > threshold:
                alpha_lock_active = True
                alpha_locked_value = float(epoch_alpha)
                print(
                    f"alpha_lock=ON, reason=val_error_rate({val_error_rate:.6f})>threshold({threshold:.6f}), "
                    f"locked_alpha={alpha_locked_value:.6f}"
                )
            elif alpha_lock_active:
                alpha_lock_active = False
                print(
                    f"alpha_lock=OFF, reason=val_error_rate({val_error_rate:.6f})<=threshold({threshold:.6f}), "
                    "alpha schedule resumes"
                )

    if current_stage == "decoder":
        attack_mode = "cropout" if use_cropout_attack else "no_crop"
        print(f"attack_mode={attack_mode}, encoder_branch=disabled")

    path_model = result_folder + "models/"
    path_encoder_decoder = path_model + "EC_" + str(epoch) + ".pth"
    path_discriminator = path_model + "D_" + str(epoch) + ".pth"
    network.save_model(path_encoder_decoder, path_discriminator)

summary_csv = os.path.join("results", "experiment_summary.csv")
header = "alpha,stage,ber_train,ber_val,psnr,ssim,locator_dice\n"
line = "{},{},{},{},{},{},{}\n".format(
    last_epoch_alpha,
    current_stage,
    last_train_avg.get("error_rate", 0.0),
    last_val_avg.get("error_rate", 0.0),
    last_val_avg.get("psnr", 0.0),
    last_val_avg.get("ssim", 0.0),
    last_val_avg.get("locator_dice", 0.0),
)

if not os.path.exists(summary_csv):
    with open(summary_csv, "w") as f:
        f.write(header)
with open(summary_csv, "a") as f:
    f.write(line)
