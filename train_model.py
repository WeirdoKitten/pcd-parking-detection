"""
Train YOLOv8 model for parking space detection
"""
from ultralytics import YOLO
import torch

def train_parking_detector():
    """
    Train YOLOv8 model on parking dataset
    """
    print("🚀 Starting YOLOv8 Training for Parking Space Detection")
    print("=" * 60)
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device.upper()}")
    
    # Load a pretrained YOLOv8 model (nano version for faster training)
    # You can use: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
    print("\n📦 Loading YOLOv8 nano model...")
    model = YOLO('yolov8n.pt')
    
    # Training parameters
    print("\n⚙️ Training Configuration:")
    epochs = 100
    imgsz = 640
    batch = 16
    patience = 20  # Early stopping patience
    
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch}")
    print(f"   Patience: {patience}")
    
    # Train the model
    print("\n🏋️ Starting training...")
    print("-" * 60)
    
    results = model.train(
        data='parking_dataset.yaml',
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        device=device,
        project='runs/parking_detection',
        name='yolov8n_parking',
        exist_ok=True,
        
        # Augmentation parameters
        hsv_h=0.015,      # Image HSV-Hue augmentation
        hsv_s=0.7,        # Image HSV-Saturation augmentation
        hsv_v=0.4,        # Image HSV-Value augmentation
        degrees=0.0,      # Rotation (keep 0 for parking lots)
        translate=0.1,    # Translation
        scale=0.5,        # Scale
        shear=0.0,        # Shear
        perspective=0.0,  # Perspective
        flipud=0.0,       # Flip up-down
        fliplr=0.5,       # Flip left-right
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.0,        # Mixup augmentation
        
        # Optimizer
        optimizer='auto',
        lr0=0.01,         # Initial learning rate
        lrf=0.01,         # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        
        # Other settings
        verbose=True,
        save=True,
        save_period=-1,   # Save checkpoint every x epochs (-1 = disabled)
        plots=True,       # Save plots
        cache=False,      # Cache images for faster training
    )
    
    print("\n" + "=" * 60)
    print("✅ Training completed!")
    print(f"📊 Best model saved at: runs/parking_detection/yolov8n_parking/weights/best.pt")
    print(f"📈 Training results saved at: runs/parking_detection/yolov8n_parking")
    
    # Validate the model
    print("\n🔍 Validating model on validation set...")
    metrics = model.val()
    
    print("\n📊 Validation Metrics:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    
    return model, results

if __name__ == '__main__':
    try:
        model, results = train_parking_detector()
        print("\n🎉 All done! Your parking detection model is ready to use.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
