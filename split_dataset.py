"""
Split dataset into train and validation sets
"""
import os
import shutil
from pathlib import Path
import random

def create_dataset_split(images_dir, labels_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Split dataset into train and validation sets
    """
    random.seed(seed)
    
    # Get all image files
    images = sorted(Path(images_dir).glob('*.png'))
    image_ids = [img.stem for img in images]
    
    # Shuffle and split
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * train_ratio)
    
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
    print(f"📊 Dataset split:")
    print(f"   Total images: {len(image_ids)}")
    print(f"   Training: {len(train_ids)} ({len(train_ids)/len(image_ids)*100:.1f}%)")
    print(f"   Validation: {len(val_ids)} ({len(val_ids)/len(image_ids)*100:.1f}%)")
    
    # Create directory structure
    output_path = Path(output_dir)
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy files
    def copy_files(ids, split):
        for img_id in ids:
            # Copy image
            src_img = Path(images_dir) / f'{img_id}.png'
            dst_img = output_path / split / 'images' / f'{img_id}.png'
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            src_label = Path(labels_dir) / f'{img_id}.txt'
            dst_label = output_path / split / 'labels' / f'{img_id}.txt'
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
        
        print(f"   ✓ Copied {len(ids)} files to {split}")
    
    print("\n🔄 Copying files...")
    copy_files(train_ids, 'train')
    copy_files(val_ids, 'val')
    
    return train_ids, val_ids

if __name__ == '__main__':
    images_dir = 'DataSet/images'
    labels_dir = 'DataSet/labels'
    output_dir = 'DataSet/yolo_dataset'
    
    train_ids, val_ids = create_dataset_split(images_dir, labels_dir, output_dir)
    
    print("\n✅ Dataset split completed!")
    print(f"   Output directory: {output_dir}")
