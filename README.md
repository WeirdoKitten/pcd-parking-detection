# 🅿️ Parking Space Detection System

> **Sistem deteksi slot parkir otomatis menggunakan YOLOv8 untuk mendeteksi dan menghitung slot parkir yang kosong, terisi, dan sebagian terisi secara real-time.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)

## 📋 Deskripsi

Proyek ini mengimplementasikan sistem Computer Vision untuk mendeteksi dan mengklasifikasikan status slot parkir secara otomatis menggunakan teknologi Deep Learning. Sistem dapat membedakan antara:
- ✅ **Slot parkir kosong** (Free parking)
- 🚗 **Slot parkir terisi** (Occupied parking)
- ⚠️ **Slot parkir sebagian terisi** (Partially free parking)

## ✨ Fitur Utama

- 🎯 Deteksi akurat dengan **mAP50: 98.4%**
- 🚀 Real-time inference menggunakan YOLOv8
- 📊 Visualisasi hasil dengan bounding box berwarna
- 📈 Statistik okupansi parkir (occupancy rate)
- 💾 Otomatis menyimpan hasil deteksi
- 🔄 Support untuk berbagai sudut pandang kamera

## 📊 Dataset

Dataset berasal dari Kaggle - Parking Space Object Detection:
- **Total gambar**: 30 images
- **Training set**: 24 images (80%)
- **Validation set**: 6 images (20%)
- **Total anotasi**: 903 slot parkir
  - Free parking: 273 slots
  - Occupied parking: 624 slots
  - Partially free: 6 slots

## 🏗️ Struktur Proyek

```
parking-detection/
├── DataSet/
│   ├── images/              # Dataset gambar asli
│   ├── boxes/               # Visualisasi bounding box
│   ├── labels/              # Anotasi format YOLO
│   ├── yolo_dataset/        # Dataset terstruktur train/val
│   ├── annotations.xml      # Anotasi XML asli
│   └── parking.csv          # Mapping file
│
├── runs/                    # Hasil training (model weights & metrics)
├── results/                 # Hasil deteksi
│
├── parse_annotations.py     # Convert XML → YOLO format
├── split_dataset.py         # Split dataset train/validation
├── parking_dataset.yaml     # Konfigurasi dataset YOLO
├── train_model.py           # Script training model
├── detect_parking.py        # Script inference/deteksi
│
├── requirements.txt         # Dependencies
└── README.md
```

## 🚀 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/WeirdoKitten/pcd-parking-detection.git
cd pcd-parking-detection
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Atau install manual:**
```bash
pip install ultralytics opencv-python pillow matplotlib
```

### 4. Persiapan Dataset

Jika Anda menggunakan dataset sendiri, jalankan preprocessing:

```bash
# Convert annotations XML ke format YOLO
python parse_annotations.py

# Split dataset menjadi train dan validation
python split_dataset.py
```

## 🎓 Training Model

### Training dari Scratch

```bash
python train_model.py
```

**Parameter Training:**
- Model: YOLOv8 Nano (yolov8n.pt)
- Epochs: 100
- Image size: 640x640
- Batch size: 16
- Early stopping: patience=20
- Augmentations: HSV, flip, mosaic, dll.

**Hasil Training:**
- Model terbaik: `runs/parking_detection/yolov8n_parking/weights/best.pt`
- Model terakhir: `runs/parking_detection/yolov8n_parking/weights/last.pt`
- Metrics & plots: `runs/parking_detection/yolov8n_parking/`

### Model Performance

| Metric | Value |
|--------|-------|
| **mAP50** | 98.4% |
| **mAP50-95** | 86.6% |
| **Precision** | 99.5% |
| **Recall** | 91.9% |

## 🔍 Testing/Inference

### Quick Test

**Langkah 1:** Buka file `detect_parking.py` dan cari baris berikut (sekitar baris 195):

```python
# Example: Test on a validation image
# You can change this to any image path
test_image = 'DataSet/yolo_dataset/test/test_3.png
```

**Langkah 2:** Ganti path gambar dengan gambar yang ingin Anda test:

```python
test_image = 'path/ke/gambar/parkir_anda.png'
# Contoh:
# test_image = 'DataSet/images/5.png'
# test_image = 'C:/Users/foto/parking_mall.jpg'
```

**Langkah 3:** Jalankan script:

```bash
python detect_parking.py
```

### Parameter Deteksi

- `image_path`: Path ke gambar parkir
- `conf_threshold`: Confidence threshold (default: 0.25)
- `save_output`: Simpan hasil visualisasi (default: True)

## 📸 Output Deteksi

Sistem menghasilkan:

1. **Visualisasi Gambar** dengan:
   - Bounding box berwarna:
     - 🟢 **Hijau** = Slot kosong
     - 🔴 **Merah** = Slot terisi
     - 🟡 **Kuning** = Slot sebagian terisi
   - Label class + confidence score
   - Statistik overlay di gambar

2. **Statistik Detail**:
   - Total parking slots
   - Jumlah slot kosong
   - Jumlah slot terisi
   - Jumlah slot sebagian terisi
   - Occupancy rate (persentase)

3. **File Output**:
   - Gambar tersimpan di folder `results/`
   - Format: `detected_[nama_file].png`

## 🛠️ Requirements

```
ultralytics>=8.0.0
opencv-python>=4.5.0
pillow>=9.0.0
matplotlib>=3.5.0
torch>=1.8.0
```

## 📁 File Penting

| File | Deskripsi |
|------|-----------|
| `parse_annotations.py` | Convert XML annotations ke format YOLO |
| `split_dataset.py` | Split dataset train/validation (80/20) |
| `train_model.py` | Training YOLOv8 model |
| `detect_parking.py` | Inference dan visualisasi hasil |
| `parking_dataset.yaml` | Konfigurasi dataset untuk YOLO |

## 🎯 Use Cases

1. **Smart Parking Management System**
   - Monitoring real-time status parkir
   - Menghitung jumlah slot tersedia
   - Menampilkan ke dashboard

2. **Parking Lot Analytics**
   - Analisis pola okupansi
   - Peak hours detection
   - Revenue optimization

3. **Mobile App Integration**
   - API untuk aplikasi parking finder
   - Notifikasi slot tersedia

## 🔧 Troubleshooting

### Error: ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### Model tidak ditemukan

Pastikan sudah menjalankan training:
```bash
python train_model.py
```

### Memory error saat training

Kurangi batch size di `train_model.py`:
```python
batch = 8  # dari 16 ke 8
```

## 🚀 Future Improvements

- [ ] Integration dengan CCTV real-time
- [ ] Web dashboard untuk monitoring
- [ ] REST API deployment
- [ ] Support untuk video processing
- [ ] Mobile app development
- [ ] Multi-camera support
- [ ] Historical data analytics

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset dari [Kaggle - Parking Space Object Detection](https://www.kaggle.com/datasets/trainingdatapro/parking-space-detection-dataset)
- YOLOv8 oleh [Ultralytics](https://github.com/ultralytics/ultralytics)
- OpenCV untuk image processing


⭐ **Jika proyek ini membantu, jangan lupa beri star!** ⭐