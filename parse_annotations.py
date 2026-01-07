"""
Parse XML annotations to YOLO format
Converts polygon annotations to bounding boxes
"""
import xml.etree.ElementTree as ET
import os
from pathlib import Path

# Label mapping
LABEL_MAP = {
    'free_parking_space': 0,
    'not_free_parking_space': 1,
    'partially_free_parking_space': 2
}

def polygon_to_bbox(points_str):
    """
    Convert polygon points to bounding box (x_center, y_center, width, height)
    Points format: "x1,y1;x2,y2;x3,y3;x4,y4"
    """
    points = []
    for point in points_str.split(';'):
        x, y = point.split(',')
        points.append((float(x), float(y)))
    
    # Get min/max coordinates
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    # Calculate bbox parameters
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height

def parse_annotations(xml_file, output_dir):
    """
    Parse annotations.xml and create YOLO format .txt files
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Create output directory for labels
    labels_dir = Path(output_dir) / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {'free': 0, 'not_free': 0, 'partially_free': 0}
    
    # Process each image
    for image in root.findall('image'):
        img_name = image.get('name')
        img_width = float(image.get('width'))
        img_height = float(image.get('height'))
        
        # Extract image filename without path and extension
        img_id = Path(img_name).stem
        
        # Create label file
        label_file = labels_dir / f'{img_id}.txt'
        
        annotations = []
        
        # Process each polygon in the image
        for polygon in image.findall('polygon'):
            label = polygon.get('label')
            points = polygon.get('points')
            
            if label in LABEL_MAP:
                # Convert polygon to bbox
                x_center, y_center, width, height = polygon_to_bbox(points)
                
                # Normalize coordinates (YOLO format requires 0-1 range)
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = width / img_width
                height_norm = height / img_height
                
                # YOLO format: class x_center y_center width height
                class_id = LABEL_MAP[label]
                annotations.append(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}")
                
                # Update statistics
                if class_id == 0:
                    stats['free'] += 1
                elif class_id == 1:
                    stats['not_free'] += 1
                elif class_id == 2:
                    stats['partially_free'] += 1
        
        # Write annotations to file
        if annotations:
            with open(label_file, 'w') as f:
                f.write('\n'.join(annotations))
            print(f"✓ Created {label_file.name} with {len(annotations)} parking slots")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Free parking spaces: {stats['free']}")
    print(f"   Not free parking spaces: {stats['not_free']}")
    print(f"   Partially free parking spaces: {stats['partially_free']}")
    print(f"   Total: {sum(stats.values())}")
    
    return stats

if __name__ == '__main__':
    # Paths
    xml_file = 'DataSet/annotations.xml'
    output_dir = 'DataSet'
    
    print("🔄 Parsing annotations from XML to YOLO format...")
    stats = parse_annotations(xml_file, output_dir)
    print("\n✅ Conversion completed!")
