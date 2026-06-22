"""
Extract physical features for new physical database.

This script extracts VideoMAE features from videos in new_physical_db.json
and saves them organized by category in subdirectories.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from decord import VideoReader, cpu
from torchvision import transforms
from PIL import Image

def get_videomae_model(model_path, videomae_root):
    """Load VideoMAE model."""
    sys.path.insert(0, str(Path(videomae_root).resolve()))
    from models import modeling_finetune

    model = modeling_finetune.vit_base_patch16_224(
        pretrained=False,
        num_classes=710,
        all_frames=16,
        tubelet_size=2,
        drop_path_rate=0.0,
        use_mean_pooling=True
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['module'], strict=False)
    model.eval()
    return model


def load_video(video_path, num_frames=16):
    """Load and sample frames from video."""
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    if total_frames == 0:
        return []
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    images = []
    for i in indices:
        images.append(Image.fromarray(vr[i].asnumpy()))
    return images


def process_images(images):
    """Process images for VideoMAE input."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensors = [transform(img) for img in images]
    # [T, C, H, W] -> [C, T, H, W] for VideoMAE
    return torch.stack(tensors).permute(1, 0, 2, 3).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Extract physical features from new physical database")
    parser.add_argument(
        "--db_json", 
        type=str, 
        default="PhysicalDB/new_physical_db.json",
        help="Path to new_physical_db.json"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="dataset/physical_features",
        help="Output directory (features will be saved in category subdirectories)"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to VideoMAE model checkpoint"
    )
    parser.add_argument(
        "--videomae_root",
        type=str,
        required=True,
        help="Path to a VideoMAEv2 source checkout",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size (currently not used, processes one video at a time)"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip videos that already have extracted features"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load database
    db_path = Path(args.db_json)
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        return
    
    with open(db_path, 'r', encoding='utf-8') as f:
        db_data = json.load(f)

    # Setup model
    model = get_videomae_model(args.model_path, args.videomae_root)
    model.to(device)
    
    # Setup output directory
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Move existing features to all_features/ subdirectory
    print("=" * 60)
    print("Step 1: Moving existing features to all_features/")
    print("=" * 60)
    
    all_features_dir = output_base / "all_features"
    all_features_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all categories from database to avoid moving category directories
    db_categories = {entry.get("category", "unknown") for entry in db_data}
    db_categories.add("all_features")  # Don't move all_features itself
    
    moved_files = 0
    moved_dirs = 0
    skipped_count = 0
    
    # Move all .pt files in the root directory
    for file_path in output_base.glob("*.pt"):
        target_path = all_features_dir / file_path.name
        if not target_path.exists():
            file_path.rename(target_path)
            moved_files += 1
        else:
            skipped_count += 1
    
    # Move all subdirectories that are not category directories
    for subdir in list(output_base.iterdir()):  # Use list() to avoid iteration issues
        if subdir.is_dir() and subdir.name not in db_categories:
            # Move all .pt files from subdirectory to all_features/
            subdir_files = list(subdir.glob("*.pt"))
            if subdir_files:
                for file_path in subdir_files:
                    target_path = all_features_dir / file_path.name
                    if not target_path.exists():
                        file_path.rename(target_path)
                        moved_files += 1
                    else:
                        skipped_count += 1
                # Remove empty subdirectory
                try:
                    if not any(subdir.iterdir()):  # Only remove if empty
                        subdir.rmdir()
                        moved_dirs += 1
                except:
                    pass
    
    print(f"✓ Moved {moved_files} files and {moved_dirs} directories to all_features/")
    if skipped_count > 0:
        print(f"⊘ Skipped {skipped_count} items (already exist)")
    print()

    # Group by category
    by_category = {}
    for entry in db_data:
        category = entry.get("category", "unknown")
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(entry)

    # Step 2: Extract features and organize by category
    print("=" * 60)
    print("Step 2: Extracting features and organizing by category")
    print("=" * 60)
    
    # Process each category
    total_processed = 0
    total_skipped = 0
    total_errors = 0

    with torch.no_grad():
        for category, entries in by_category.items():
            # Create category subdirectory
            category_dir = output_base / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Process videos in this category
            for entry in tqdm(entries, desc=f"Extracting {category}"):
                video_path = Path(entry["video_path"])
                video_name = entry["video_name"]
                
                # Output path: category_dir/video_name.pt
                save_path = category_dir / (Path(video_name).stem + ".pt")
                
                # Skip if exists
                if args.skip_existing and save_path.exists():
                    total_skipped += 1
                    continue
                
                # Check if video exists
                if not video_path.exists():
                    total_errors += 1
                    continue
                
                try:
                    # Load and process video
                    images = load_video(video_path)
                    if len(images) == 0:
                        total_errors += 1
                        continue
                    
                    inputs = process_images(images).to(device)  # [1, C, T, H, W]
                    
                    # Forward pass to get features
                    features = model.forward_features(inputs)  # [B, N, C]
                    
                    # Save features
                    torch.save(features.cpu(), save_path)
                    total_processed += 1
                    
                except Exception as e:
                    total_errors += 1
                    continue

    # Summary
    print(f"\n{'='*60}")
    print("Extraction Summary:")
    print(f"{'='*60}")
    print(f"✓ Successfully processed: {total_processed}")
    print(f"⊘ Skipped (already exists): {total_skipped}")
    print(f"❌ Errors: {total_errors}")
    print(f"📁 Output directory: {output_base}")
    print(f"📂 Categories: {list(by_category.keys())}")


if __name__ == "__main__":
    main()
