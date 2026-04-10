#!/usr/bin/env python3
"""
Checkpoint Selector — Eval-based Best Checkpoint Selection

Setelah training SDXL selesai, script ini mengevaluasi semua checkpoint
menggunakan img2img reconstruction loss (L2 pixel MSE) yang sama dengan
evaluator G.O.D, lalu memilih checkpoint dengan loss terendah.

Logic evaluator:
- Load base model + LoRA
- img2img pada test images (denoise=0.9, cfg=8, steps=20)
- L2 pixel loss = np.mean((original - generated) ** 2)
- No-text mode (prompt kosong) = 75% weight di evaluator
- Kita fokus ke no-text mode saja (dominant weight)

Usage:
    python checkpoint_selector.py \
        --output-dir /path/to/checkpoints \
        --base-model /path/to/base_model \
        --train-data-dir /path/to/training/images \
        --model-type sdxl
"""

import argparse
import glob
import os
import random
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# === Eval Parameters (MUST match G.O.D validator/core/constants.py) ===
EVAL_PARAMS = {
    "sdxl": {"steps": 20, "cfg": 8.0, "denoise": 0.9},
    "flux": {"steps": 35, "cfg": 100.0, "denoise": 0.75},
}

DEFAULT_NUM_SEEDS = 2
DEFAULT_MAX_IMAGES = 5
DEFAULT_MAX_TIME_SECONDS = 900  # 15 menit max


def get_holdout_count(total_images: int) -> int:
    """
    Tentukan jumlah gambar yang disisihkan sebagai hold-out set.
    
    Rules:
    - 1-10 images  → 1
    - 11-20 images → 2
    - 21-30 images → 3
    - 31-40 images → 4
    - 41-50 images → 5
    - >50 images   → 5 (fixed max)
    """
    if total_images <= 0:
        return 0
    return min(5, max(1, (total_images - 1) // 10 + 1))


def create_holdout_set(train_data_dir: str, holdout_dir: str | None = None) -> str | None:
    """
    Sisihkan beberapa gambar dari training set sebagai pseudo-test set.
    
    Gambar dipindahkan (move, bukan copy) dari subdirectory training ke
    holdout_dir supaya tidak ikut training. Ini memastikan checkpoint
    selector evaluasi pakai unseen images, mirip evaluator G.O.D.
    
    Args:
        train_data_dir: Path ke training images (berisi subdirectory seperti 5_lora person/)
        holdout_dir: Target directory untuk hold-out images. Default: train_data_dir + "_holdout"
    
    Returns:
        Path ke holdout directory, atau None jika gagal
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    # Kumpulkan semua gambar dari semua subdirectory
    all_images = []
    for root, _, files in os.walk(train_data_dir):
        for f in files:
            if f.startswith("."):
                continue
            _, ext = os.path.splitext(f.lower())
            if ext in image_extensions:
                all_images.append(os.path.join(root, f))
    
    total = len(all_images)
    if total <= 1:
        print(f"⚠️  Hanya {total} gambar di training set, skip hold-out.", flush=True)
        return None
    
    holdout_count = get_holdout_count(total)
    
    # Pastikan masih ada cukup gambar untuk training
    if total - holdout_count < 1:
        holdout_count = max(1, total - 1)
    
    if holdout_dir is None:
        holdout_dir = train_data_dir.rstrip("/") + "_holdout"
    
    os.makedirs(holdout_dir, exist_ok=True)
    
    # Random select with fixed seed for reproducibility
    random.seed(42)
    holdout_images = random.sample(all_images, holdout_count)
    
    print(f"\n📦 HOLD-OUT SET", flush=True)
    print(f"   Total images: {total}", flush=True)
    print(f"   Hold-out: {holdout_count} gambar", flush=True)
    print(f"   Training: {total - holdout_count} gambar", flush=True)
    
    for img_path in holdout_images:
        basename = os.path.basename(img_path)
        dest = os.path.join(holdout_dir, basename)
        
        # Juga pindahkan file .txt caption jika ada
        name_without_ext = os.path.splitext(basename)[0]
        caption_path = os.path.join(os.path.dirname(img_path), f"{name_without_ext}.txt")
        
        shutil.move(img_path, dest)
        print(f"   Moved: {basename} → holdout/", flush=True)
        
        if os.path.exists(caption_path):
            caption_dest = os.path.join(holdout_dir, f"{name_without_ext}.txt")
            shutil.move(caption_path, caption_dest)
    
    print(f"   📁 Holdout dir: {holdout_dir}", flush=True)
    return holdout_dir


def find_checkpoints(output_dir: str) -> list[dict]:
    """
    Cari semua LoRA checkpoint di output_dir.
    
    sd-scripts naming convention:
    - last.safetensors (final checkpoint)
    - last-000005.safetensors (epoch 5 checkpoint)
    - last-000010.safetensors (epoch 10 checkpoint)
    """
    checkpoints = []
    
    for f in sorted(glob.glob(os.path.join(output_dir, "*.safetensors"))):
        basename = os.path.basename(f)
        
        # Parse epoch number
        if basename == "last.safetensors":
            epoch = float("inf")  # Final checkpoint
            label = "last (final)"
        elif basename.startswith("last-") and basename.endswith(".safetensors"):
            try:
                epoch_str = basename.replace("last-", "").replace(".safetensors", "")
                epoch = int(epoch_str)
                label = f"epoch-{epoch}"
            except ValueError:
                continue
        else:
            # Unknown naming convention, still try
            epoch = 0
            label = basename
        
        checkpoints.append({
            "path": f,
            "basename": basename,
            "epoch": epoch,
            "label": label,
        })
    
    return checkpoints


def get_eval_images(eval_dir: str, max_images: int = 5) -> list[str]:
    """
    Ambil gambar untuk evaluasi dari directory yang diberikan.
    
    Bisa dari holdout_dir (preferred) atau train_data_dir (fallback).
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = []
    
    for root, _, files in os.walk(eval_dir):
        for f in files:
            if f.startswith("."):
                continue
            _, ext = os.path.splitext(f.lower())
            if ext in image_extensions:
                image_files.append(os.path.join(root, f))
    
    if not image_files:
        raise FileNotFoundError(f"Tidak ada gambar ditemukan di {eval_dir}")
    
    # Random subset
    random.seed(42)  # Reproducible
    if len(image_files) > max_images:
        image_files = random.sample(image_files, max_images)
    
    print(f"📸 Found {len(image_files)} eval images from: {eval_dir}", flush=True)
    return image_files


def calculate_l2_loss(original: np.ndarray, generated: np.ndarray) -> float:
    """
    L2 pixel loss — sama persis dengan evaluator.
    
    Dari G.O.D/validator/evaluation/eval_diffusion.py:
        test_image = np.array(test_image.convert("RGB")) / 255.0
        generated_image = np.array(generated_image.convert("RGB")) / 255.0
        l2_loss = np.mean((test_image - generated_image) ** 2)
    """
    return float(np.mean((original - generated) ** 2))


def evaluate_checkpoint(
    pipe,
    lora_path: str,
    image_files: list[str],
    params: dict,
    num_seeds: int = 2,
    device: str = "cuda",
) -> float:
    """
    Evaluate satu LoRA checkpoint.
    
    Returns:
        Average L2 loss (lower is better)
    """
    from PIL import Image
    
    # Load LoRA
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass
    
    pipe.load_lora_weights(lora_path)
    
    # Generate seeds
    random.seed(42)
    seeds = [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]
    
    all_losses = []
    
    for img_path in image_files:
        # Load and prep image
        image = Image.open(img_path).convert("RGB")
        
        # Resize to 1024x1024 (eval resolution)
        width, height = image.size
        target_size = 1024
        
        # Adjust to nearest multiple of 8 that fits within 1024
        if width != target_size or height != target_size:
            ratio = min(target_size / width, target_size / height)
            new_w = int(width * ratio) // 8 * 8
            new_h = int(height * ratio) // 8 * 8
            new_w = max(new_w, 8)
            new_h = max(new_h, 8)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        
        original_np = np.array(image).astype(np.float32) / 255.0
        
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            
            # img2img — no text mode (empty prompt, 75% of eval weight)
            with torch.no_grad():
                output = pipe(
                    prompt="",
                    image=image,
                    strength=params["denoise"],
                    guidance_scale=params["cfg"],
                    num_inference_steps=params["steps"],
                    generator=generator,
                ).images[0]
            
            generated_np = np.array(output.resize(image.size, Image.LANCZOS)).astype(np.float32) / 255.0
            
            loss = calculate_l2_loss(original_np, generated_np)
            all_losses.append(loss)
    
    avg_loss = float(np.mean(all_losses))
    return avg_loss


def load_pipeline(base_model_path: str, model_type: str, device: str = "cuda"):
    """
    Load SDXL img2img pipeline.
    
    Supports both:
    - Single safetensors file
    - Diffusers directory format
    """
    from diffusers import StableDiffusionXLImg2ImgPipeline, AutoPipelineForImage2Image

    print(f"🔄 Loading base model: {base_model_path}", flush=True)
    
    if base_model_path.endswith(".safetensors"):
        # Single file — load via from_single_file
        pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            base_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    else:
        # Diffusers format directory
        pipe = AutoPipelineForImage2Image.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
    
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Memory optimization
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    
    print("✅ Pipeline loaded", flush=True)
    return pipe


def select_best_checkpoint(
    output_dir: str,
    base_model_path: str,
    train_data_dir: str,
    model_type: str = "sdxl",
    max_images: int = DEFAULT_MAX_IMAGES,
    num_seeds: int = DEFAULT_NUM_SEEDS,
    max_time: int = DEFAULT_MAX_TIME_SECONDS,
    dry_run: bool = False,
    device: str = "cuda",
    holdout_dir: str | None = None,
):
    """
    Main function — evaluate all checkpoints and select the best one.
    
    Args:
        output_dir: Directory containing checkpoint .safetensors files
        base_model_path: Path to base SDXL model
        train_data_dir: Path to training images (fallback if no holdout)
        model_type: Model type (sdxl)
        max_images: Max images to use for evaluation
        num_seeds: Seeds per image
        max_time: Max total time in seconds
        dry_run: Only show what would be done
        device: CUDA device
        holdout_dir: Path to hold-out images (preferred over train_data_dir)
    """
    print("\n" + "=" * 60, flush=True)
    print("🏆 CHECKPOINT SELECTOR", flush=True)
    print("=" * 60, flush=True)
    
    start_time = time.time()
    
    # 1. Find checkpoints
    checkpoints = find_checkpoints(output_dir)
    
    if len(checkpoints) <= 1:
        print("⚠️  Hanya ada 1 checkpoint atau tidak ada. Skip selection.", flush=True)
        return
    
    print(f"\n📁 Found {len(checkpoints)} checkpoints:", flush=True)
    for cp in checkpoints:
        print(f"   - {cp['label']}: {cp['basename']}", flush=True)
    
    # 2. Get evaluation images (prefer holdout, fallback to training)
    eval_source = holdout_dir if holdout_dir and os.path.isdir(holdout_dir) else train_data_dir
    if eval_source == holdout_dir:
        print(f"\n✅ Menggunakan HOLD-OUT images (unseen data) untuk evaluasi", flush=True)
    else:
        print(f"\n⚠️  Menggunakan TRAINING images untuk evaluasi (no holdout available)", flush=True)
    image_files = get_eval_images(eval_source, max_images)
    
    if dry_run:
        print("\n🔍 DRY RUN — showing plan only", flush=True)
        print(f"   Would evaluate {len(checkpoints)} checkpoints", flush=True)
        print(f"   Using {len(image_files)} images × {num_seeds} seeds", flush=True)
        print(f"   Total inferences: {len(checkpoints) * len(image_files) * num_seeds}", flush=True)
        return
    
    # 3. Get eval parameters
    params = EVAL_PARAMS.get(model_type, EVAL_PARAMS["sdxl"])
    print(f"\n⚙️  Eval params: steps={params['steps']}, cfg={params['cfg']}, denoise={params['denoise']}", flush=True)
    
    # 4. Load pipeline
    pipe = load_pipeline(base_model_path, model_type, device)
    
    # 5. Evaluate each checkpoint
    results = []
    
    for i, cp in enumerate(checkpoints):
        elapsed = time.time() - start_time
        if elapsed > max_time:
            print(f"\n⏰ Timeout ({max_time}s). Evaluasi dihentikan setelah {i} checkpoints.", flush=True)
            break
        
        print(f"\n📊 [{i+1}/{len(checkpoints)}] Evaluating: {cp['label']}...", flush=True)
        
        try:
            loss = evaluate_checkpoint(
                pipe=pipe,
                lora_path=cp["path"],
                image_files=image_files,
                params=params,
                num_seeds=num_seeds,
                device=device,
            )
            
            results.append({**cp, "loss": loss})
            print(f"   Loss: {loss:.6f}", flush=True)
            
        except Exception as e:
            print(f"   ❌ Error: {e}", flush=True)
            continue
    
    # 6. Select best
    if not results:
        print("\n⚠️  Tidak ada checkpoint yang berhasil dievaluasi. Tetap pakai last.", flush=True)
        return
    
    results.sort(key=lambda x: x["loss"])
    best = results[0]
    
    print("\n" + "-" * 60, flush=True)
    print("📋 RESULTS (sorted by loss, lower is better):", flush=True)
    print("-" * 60, flush=True)
    
    for i, r in enumerate(results):
        marker = " 🏆" if i == 0 else ""
        print(f"   {r['label']:20s} | Loss: {r['loss']:.6f}{marker}", flush=True)
    
    # 7. Copy best → last.safetensors
    last_path = os.path.join(output_dir, "last.safetensors")
    
    if best["basename"] == "last.safetensors":
        print(f"\n✅ Best checkpoint sudah 'last.safetensors'. Tidak perlu copy.", flush=True)
    else:
        print(f"\n🔄 Replacing last.safetensors dengan {best['label']}...", flush=True)
        
        # Backup original last
        backup_path = os.path.join(output_dir, "last_original_backup.safetensors")
        if os.path.exists(last_path):
            shutil.copy2(last_path, backup_path)
            print(f"   Backup: {backup_path}", flush=True)
        
        # Copy best → last.safetensors
        shutil.copy2(best["path"], last_path)
        print(f"   ✅ Copied {best['basename']} → last.safetensors", flush=True)
    
    # 8. Cleanup pipeline
    del pipe
    torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Eval-based checkpoint selector for SDXL LoRA training"
    )
    parser.add_argument("--output-dir", required=True, help="Directory containing checkpoint safetensors files")
    parser.add_argument("--base-model", required=True, help="Path to base SDXL model (safetensors or diffusers dir)")
    parser.add_argument("--train-data-dir", required=True, help="Path to training images directory")
    parser.add_argument("--model-type", default="sdxl", choices=["sdxl"], help="Model type")
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES, help="Max images for evaluation")
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS, help="Seeds per image")
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_TIME_SECONDS, help="Max total time in seconds")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be done")
    
    args = parser.parse_args()
    
    select_best_checkpoint(
        output_dir=args.output_dir,
        base_model_path=args.base_model,
        train_data_dir=args.train_data_dir,
        model_type=args.model_type,
        max_images=args.max_images,
        num_seeds=args.num_seeds,
        max_time=args.max_time,
        dry_run=args.dry_run,
        device=args.device,
    )


if __name__ == "__main__":
    main()
