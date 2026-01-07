"""
Parking Space Detection - Web UI
Simple interface untuk upload dan deteksi slot parkir
"""
import gradio as gr
from detect_parking import ParkingDetector
from pathlib import Path
import cv2
import numpy as np

# Initialize detector
MODEL_PATH = 'runs/parking_detection/yolov8n_parking/weights/best.pt'

def check_model_exists():
    """Check if model exists"""
    if not Path(MODEL_PATH).exists():
        return False, f"❌ Model tidak ditemukan di: {MODEL_PATH}\n\nSilakan training model terlebih dahulu dengan menjalankan:\npython train_model.py"
    return True, "Model loaded successfully"

def detect_parking_spaces(image):
    """
    Detect parking spaces from uploaded image
    
    Args:
        image: numpy array dari gambar yang diupload
    
    Returns:
        annotated_image: gambar dengan bounding boxes
        statistics: text dengan statistik
    """
    # Check if model exists
    model_exists, message = check_model_exists()
    if not model_exists:
        return None, message
    
    try:
        # Initialize detector
        detector = ParkingDetector(MODEL_PATH)
        
        # Save temporary image
        temp_path = 'temp_upload.png'
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Run detection with default confidence
        results = detector.detect(temp_path, conf_threshold=0.3, save_output=False)
        
        # Get annotated image
        annotated_image = results['image']
        
        # Create statistics text
        total = results['total_slots']
        free = results['free']
        occupied = results['occupied']
        partially_free = results['partially_free']
        occupancy = results['occupancy_rate']
        
        stats_text = f"""
## 📊 Hasil Deteksi Parking

### Statistik Slot Parkir:
- 🅿️ **Total Slot Parkir**: {total}
- ✅ **Slot Kosong**: {free}
- 🚗 **Slot Terisi**: {occupied}
- ⚠️ **Slot Sebagian Terisi**: {partially_free}

### Tingkat Okupansi:
- 📈 **Occupancy Rate**: {occupancy:.1f}%
- 📉 **Availability Rate**: {100-occupancy:.1f}%

---
### Keterangan Warna:
- 🟢 **Hijau** = Slot Kosong (Free)
- 🔴 **Merah** = Slot Terisi (Occupied)
- 🟡 **Kuning** = Slot Sebagian Terisi (Partially Free)
        """
        
        # Clean up
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        
        return annotated_image, stats_text
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}\n\nPastikan gambar valid dan model sudah terlatih."

# Create Gradio interface
with gr.Blocks(title="🅿️ Parking Detection System") as app:
    
    app.theme = gr.themes.Soft()
    
    gr.Markdown("""
    # 🅿️ Parking Space Detection System
    
    Upload gambar parkiran untuk mendeteksi slot yang kosong dan terisi secara otomatis!
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 Upload Gambar Parkiran")
            image_input = gr.Image(
                label="Pilih Gambar",
                type="numpy",
                height=450
            )
            
            detect_btn = gr.Button("🔍 Deteksi Slot Parkir", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            ### 📝 Cara Menggunakan:
            1. Upload gambar parkiran (PNG, JPG, dll)
            2. Klik tombol "Deteksi Slot Parkir"
            3. Lihat hasil deteksi dan statistik
            """)
        
        with gr.Column():
            gr.Markdown("### 📊 Hasil Deteksi")
            image_output = gr.Image(
                label="Gambar dengan Bounding Box",
                height=450
            )
            
            stats_output = gr.Markdown(
                value="📤 Upload gambar dan klik 'Deteksi Slot Parkir' untuk melihat hasil."
            )
    
    # Footer
    gr.Markdown("""
    ---
    ### 🎯 Keterangan Warna:
    - 🟢 **Hijau** = Slot Kosong (Free)
    - 🔴 **Merah** = Slot Terisi (Occupied)
    - 🟡 **Kuning** = Slot Sebagian Terisi (Partially Free)
    
    ### 📊 Model Performance:
    **mAP50**: 98.4% | **Precision**: 99.5% | **Recall**: 91.9%
    
    ---
    © 2026 Parking Detection System - Powered by YOLOv8
    """)
    
    # Connect button to function
    detect_btn.click(
        fn=detect_parking_spaces,
        inputs=[image_input],
        outputs=[image_output, stats_output]
    )

# Launch app
if __name__ == "__main__":
    # Check if model exists before launching
    model_exists, message = check_model_exists()
    
    if not model_exists:
        print("\n" + "="*60)
        print(message)
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("🚀 Launching Parking Detection System...")
        print("="*60 + "\n")
        
        # Launch Gradio app
        app.launch(
            share=False,  # Set True untuk generate public link
            server_name="127.0.0.1",  # localhost only
            server_port=7861,
            show_error=True,
            inbrowser=True  # Auto open browser
        )
