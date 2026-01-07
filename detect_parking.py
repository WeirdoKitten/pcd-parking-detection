"""
Parking Space Detection - Inference Script
Upload an image and detect free/occupied parking spaces
"""
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class ParkingDetector:
    def __init__(self, model_path='runs/parking_detection/yolov8n_parking/weights/best.pt'):
        """
        Initialize parking detector
        
        Args:
            model_path: Path to trained YOLO model
        """
        print(f"📦 Loading model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Class names and colors
        self.class_names = {
            0: 'Free',
            1: 'Occupied', 
            2: 'Partially Free'
        }
        
        self.colors = {
            0: (0, 255, 0),      # Green for free
            1: (0, 0, 255),      # Red for occupied
            2: (0, 255, 255)     # Yellow for partially free
        }
        
        print("✅ Model loaded successfully!")
    
    def detect(self, image_path, conf_threshold=0.25, save_output=True):
        """
        Detect parking spaces in an image
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            save_output: Whether to save annotated image
            
        Returns:
            Dictionary with detection results and statistics
        """
        print(f"\n🔍 Detecting parking spaces in: {image_path}")
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            save=False,
            verbose=False
        )
        
        result = results[0]
        
        # Read original image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Count detections by class
        counts = {
            'free': 0,
            'occupied': 0,
            'partially_free': 0
        }
        
        # Process detections
        boxes = result.boxes
        
        if len(boxes) > 0:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Update counts
                if cls == 0:
                    counts['free'] += 1
                elif cls == 1:
                    counts['occupied'] += 1
                elif cls == 2:
                    counts['partially_free'] += 1
                
                # Draw bounding box
                color = self.colors[cls]
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{self.class_names[cls]} {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # Label background
                cv2.rectangle(img_rgb, 
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color, -1)
                
                # Label text
                cv2.putText(img_rgb, label,
                          (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 255, 255), 2)
        
        # Calculate statistics
        total_slots = sum(counts.values())
        occupancy_rate = (counts['occupied'] / total_slots * 100) if total_slots > 0 else 0
        
        # Add statistics overlay
        stats_text = [
            f"Total Slots: {total_slots}",
            f"Free: {counts['free']}",
            f"Occupied: {counts['occupied']}",
            f"Partially Free: {counts['partially_free']}",
            f"Occupancy Rate: {occupancy_rate:.1f}%"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(img_rgb, text,
                      (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img_rgb, text,
                      (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.7, (0, 0, 0), 1, cv2.LINE_AA)
            y_offset += 30
        
        # Save output
        if save_output:
            output_path = Path('results') / f"detected_{Path(image_path).name}"
            output_path.parent.mkdir(exist_ok=True)
            
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), img_bgr)
            print(f"💾 Saved result to: {output_path}")
        
        # Print results
        print("\n" + "=" * 50)
        print("📊 DETECTION RESULTS")
        print("=" * 50)
        print(f"🅿️  Total Parking Slots: {total_slots}")
        print(f"✅ Free Slots: {counts['free']}")
        print(f"🚗 Occupied Slots: {counts['occupied']}")
        print(f"⚠️  Partially Free Slots: {counts['partially_free']}")
        print(f"📈 Occupancy Rate: {occupancy_rate:.1f}%")
        print("=" * 50)
        
        # Display image
        plt.figure(figsize=(15, 10))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f'Parking Detection Results - {total_slots} slots detected', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return {
            'total_slots': total_slots,
            'free': counts['free'],
            'occupied': counts['occupied'],
            'partially_free': counts['partially_free'],
            'occupancy_rate': occupancy_rate,
            'image': img_rgb
        }

def main():
    """
    Main function for testing the detector
    """
    print("=" * 60)
    print("🅿️  PARKING SPACE DETECTION SYSTEM")
    print("=" * 60)
    
    # Check if model exists
    model_path = 'runs/parking_detection/yolov8n_parking/weights/best.pt'
    if not Path(model_path).exists():
        print(f"\n❌ Model not found at: {model_path}")
        print("Please train the model first by running: python train_model.py")
        return
    
    # Initialize detector
    detector = ParkingDetector(model_path)
    
    # Example: Test on a validation image
    # You can change this to any image path
    test_image = 'DataSet/yolo_dataset/test/test_6.png'
    
    if Path(test_image).exists():
        results = detector.detect(test_image, conf_threshold=0.3)
    else:
        print(f"\n⚠️  Test image not found: {test_image}")
        print("Please provide a valid image path")
        print("\nUsage:")
        print("   detector = ParkingDetector()")
        print("   results = detector.detect('path/to/image.png')")

if __name__ == '__main__':
    main()
