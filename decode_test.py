"""
解码测试脚本 - 每个文件夹单独保存结果
"""
import os
import json
import random
import csv
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from network.Network import Network

WATERMARK_BITS = "100011100111000011011011010010100101011101110010"

CROPS_COUNT = 21
CROP_SIZE = 128
TILE_SIZE = 64

MODEL_PATH = r"./saved_model/EC_100.pth"

TEST_FOLDERS = [
    r"E:\projects\datasets_10000\datasets_10000\test_test\equipment\iphone15_screen1_rotated_right_crop_400_png",
    r"E:\projects\datasets_10000\datasets_10000\test_test\equipment\iphone15_screen2_crop_400_png",
]

OUTPUT_ROOT = r"E:\projects\Proposed - 副本\decode_results_all"

class ScreenCaptureDataset(Dataset):
    """屏摄数据集 - 加载文件夹中所有图片"""

    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_paths = []

        if not os.path.exists(folder):
            print(f"[警告] 文件夹不存在: {folder}")
            return

        for name in os.listdir(folder):
            file_path = os.path.join(folder, name)
            if os.path.isfile(file_path):
                try:
                    with Image.open(file_path) as img:
                        if img.size[0] > 0 and img.size[1] > 0:
                            self.image_paths.append(file_path)
                except Exception:
                    continue

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        pil = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(pil)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            image = transform(pil)

        return image, image_path

def decode_with_vote(net, image, n_crops=21, crop_size=128):
    """
    使用投票机制解码
    """
    device = next(net.encoder_decoder.parameters()).device
    _, _, h, w = image.shape

    coords = []
    max_top = max(0, h - crop_size)
    max_left = max(0, w - crop_size)

    for _ in range(n_crops):
        top = int(torch.randint(0, max_top + 1, (1,)).item())
        left = int(torch.randint(0, max_left + 1, (1,)).item())
        coords.append((top, left))

    probs_list = []
    with torch.no_grad():
        for top, left in coords:
            patch = image[:, :, top:top + crop_size, left:left + crop_size]

            if patch.shape[2] != crop_size or patch.shape[3] != crop_size:
                continue

            locator_logit = net.encoder_decoder.module.decoder.forward_locator(patch)
            corrected, _ = net._crop_from_locator(patch, locator_logit, offsets_embed=None)
            logit = net.encoder_decoder.module.decoder.forward_decode(corrected)
            prob = torch.sigmoid(logit).squeeze(0)
            probs_list.append(prob.cpu())

    if len(probs_list) == 0:
        return None

    probs = torch.stack(probs_list, dim=0)
    return probs

def decode_folder(net, folder, watermark, message_length, output_dir):
    """
    解码单个文件夹并保存结果
    """
    device = next(net.encoder_decoder.parameters()).device

    os.makedirs(output_dir, exist_ok=True)

    dataset = ScreenCaptureDataset(folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    if len(dataset) == 0:
        print(f"  跳过空文件夹: {folder}")
        return None

    print(f"  图片数量: {len(dataset)}")

    results = []
    all_correct_bits = 0
    all_total_bits = 0

    for idx, (image, image_path) in enumerate(dataloader):
        image = image.to(device)
        image_path_str = image_path[0]

        probs = decode_with_vote(net, image, n_crops=CROPS_COUNT, crop_size=CROP_SIZE)

        if probs is None:
            print(f"    [{idx+1}/{len(dataset)}] 解码失败")
            continue

        mean_prob = probs.mean(dim=0)
        pred_bits = (mean_prob >= 0.5).float().to(device)

        gt = watermark.squeeze(0)
        errors = (pred_bits != gt).float()
        error_rate = errors.mean().item()
        correct_bits = (pred_bits == gt).sum().item()

        all_correct_bits += correct_bits
        all_total_bits += message_length

        results.append({
            "filename": os.path.basename(image_path_str),
            "error_rate": error_rate,
            "correct_bits": correct_bits,
            "total_bits": message_length,
        })

    folder_ber = 1 - (all_correct_bits / all_total_bits) if all_total_bits > 0 else 1.0

    log_file = os.path.join(output_dir, "decode_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"解码日志 - {os.path.basename(folder)}\n")
        f.write(f"模型: {MODEL_PATH}\n")
        f.write(f"水印: {WATERMARK_BITS}\n")
        f.write(f"裁剪份数: {CROPS_COUNT}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"图片总数: {len(results)}\n")
        f.write(f"总体 BER: {folder_ber:.6f}\n")
        f.write(f"正确率: {1 - folder_ber:.6f}\n")

        f.write("\n----------- 详细结果 -----------\n")
        for r in results:
            f.write(f"{r['filename']}: BER={r['error_rate']:.4f} ({r['correct_bits']}/{r['total_bits']})\n")

    csv_file = os.path.join(output_dir, "decode_results.csv")
    with open(csv_file, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "error_rate", "correct_bits", "total_bits"])
        for r in results:
            writer.writerow([r["filename"], r["error_rate"], r["correct_bits"], r["total_bits"]])

    print(f"  BER: {folder_ber:.4f}, 结果已保存到: {output_dir}")

    return {
        "folder": folder,
        "folder_name": os.path.basename(os.path.dirname(folder)),
        "image_count": len(results),
        "ber": folder_ber,
        "accuracy": 1 - folder_ber,
    }

def main():

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    watermark_np = np.array([int(b) for b in WATERMARK_BITS], dtype=np.float32)
    watermark = torch.from_numpy(watermark_np).unsqueeze(0).to(device)
    message_length = len(WATERMARK_BITS)
    print(f"水印长度: {message_length} bits")
    print(f"水印信息: {WATERMARK_BITS}")

    print(f"\n加载模型: {MODEL_PATH}")

    H, W = 192, 192

    net = Network(
        H=H,
        W=W,
        message_length=message_length,
        noise_layers=["Identity()"],
        device=device,
        batch_size=1,
        lr=1e-3,
        with_diffusion=False,
        decoder_weight=1.0,
        grad_clip_norm=0.0,
        alpha=0.032,
        locator_pretrain=False,
        single_tile_embed=False,
        random_tile_position=False,
        locator_feature_supervision=False,
        locator_feat_weight=1.0,
        locator_crop_size=128,
        locator_threshold=0.5,
        inputs_preprocessed=False,
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

    net.load_model_ed(MODEL_PATH)
    net.encoder_decoder.eval()
    print("模型加载成功!")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f"\n汇总结果将保存到: {OUTPUT_ROOT}")

    summary_results = []

    print("\n" + "=" * 80)
    print(f"开始解码 {len(TEST_FOLDERS)} 个文件夹")
    print("=" * 80)

    for i, folder in enumerate(TEST_FOLDERS):
        folder_name = os.path.basename(folder)
        print(f"\n[{i+1}/{len(TEST_FOLDERS)}] 处理文件夹: {folder_name}")

        folder_output = os.path.join(OUTPUT_ROOT, folder_name)

        result = decode_folder(net, folder, watermark, message_length, folder_output)

        if result is not None:
            summary_results.append(result)

    summary_csv = os.path.join(OUTPUT_ROOT, "summary_results.csv")
    with open(summary_csv, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "image_count", "ber", "accuracy"])
        for r in summary_results:
            writer.writerow([r["folder_name"], r["image_count"], r["ber"], r["accuracy"]])

    summary_report = os.path.join(OUTPUT_ROOT, "summary_report.txt")
    with open(summary_report, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("解码测试汇总报告\n")
        f.write(f"模型: {MODEL_PATH}\n")
        f.write(f"水印: {WATERMARK_BITS}\n")
        f.write(f"裁剪份数: {CROPS_COUNT}\n")
        f.write("=" * 80 + "\n\n")

        f.write("----------- 各文件夹结果 -----------\n")
        f.write(f"{'文件夹':<20} {'图片数':<8} {'BER':<10} {'正确率':<10}\n")
        f.write("-" * 50 + "\n")

        total_images = 0
        total_errors = 0
        total_bits = 0

        for r in summary_results:
            f.write(f"{r['folder_name']:<20} {r['image_count']:<8} {r['ber']:<10.6f} {r['accuracy']:<10.6f}\n")
            total_images += r["image_count"]
            total_errors += r["ber"] * r["image_count"] * message_length
            total_bits += r["image_count"] * message_length

        overall_ber = total_errors / total_bits if total_bits > 0 else 1.0

        f.write("\n----------- 总体结果 -----------\n")
        f.write(f"总图片数: {total_images}\n")
        f.write(f"总体 BER: {overall_ber:.6f}\n")
        f.write(f"总体正确率: {1 - overall_ber:.6f}\n")

    print("\n" + "=" * 80)
    print("解码完成!")
    print(f"汇总结果保存到: {OUTPUT_ROOT}")
    print(f"总体 BER: {overall_ber:.6f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
