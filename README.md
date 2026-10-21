# 🔢 MNIST Digit Recognizer - CNN Web App

Web app nhận diện chữ số viết tay sử dụng Convolutional Neural Network (CNN) được train trên bộ dữ liệu MNIST.

## 🎯 Tính Năng
- ✅ Vẽ chữ số trực tiếp trên canvas
- ✅ Dự đoán realtime (auto sau 1.5 giây)
- ✅ Hiển thị xác suất 10 chữ số
- ✅ Responsive - hỗ trợ cả Desktop và Mobile
- ✅ CNN accuracy > 99% trên MNIST test set

## 📁 Cấu Trúc Project
```
CNN_SoVietTay/
├── app.py                      # Flask application
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render deployment config
├── Procfile                    # Process file cho Gunicorn
├── README.md                   # Tài liệu này
├── cnn_mnist_training.ipynb    # Jupyter notebook huấn luyện CNN
├── data/
│   ├── mnist_train.csv         # 60,000 mẫu train
│   └── mnist_test.csv          # 10,000 mẫu test
├── artifacts/
│   ├── cnn_mnist_model.keras   # Model đã train
│   └── model_metadata.json    # Metadata mô hình
├── static/
│   ├── css/style.css
│   └── js/canvas.js
└── templates/
    └── index.html              # Giao diện web
```

## 🚀 Chạy Local

### Bước 1: Clone repository
```bash
git clone <your-repo-url>
cd CNN_SoVietTay
```

### Bước 2: Cài thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Train mô hình
Mở và chạy notebook `cnn_mnist_training.ipynb` để train và lưu mô hình vào `artifacts/`.

### Bước 4: Chạy Flask
```bash
python app.py
```

Truy cập: http://localhost:5000

## ☁️ Deploy Lên Render (Thủ Công)

### Bước 1: Chuẩn bị
- Tạo tài khoản tại [render.com](https://render.com)
- Upload model `cnn_mnist_model.keras` vào thư mục `artifacts/` trong repository

> ⚠️ **Lưu ý quan trọng**: File model `.keras` phải được commit vào Git hoặc upload lên Render Disk vì data/*.csv được gitignore.

### Bước 2: Tạo GitHub Repository
```bash
git init
git add .
git commit -m "Initial commit: CNN MNIST digit recognizer"
git branch -M main
git remote add origin https://github.com/username/mnist-cnn.git
git push -u origin main
```

### Bước 3: Tạo Web Service trên Render
1. Đăng nhập [dashboard.render.com](https://dashboard.render.com)
2. Click **"+ New"** → **"Web Service"**
3. Connect GitHub repository của bạn
4. Cấu hình:
   - **Name**: `mnist-digit-recognizer`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT`
   - **Plan**: `Free`
5. Click **"Create Web Service"**

### Bước 4: Cấu hình Environment Variables
Trong Render Dashboard → Environment tab:
```
TF_CPP_MIN_LOG_LEVEL = 3
TF_ENABLE_ONEDNN_OPTS = 0
```

### Bước 5: Upload Model File
Vì model file lớn (>25MB), có 2 cách:

**Cách A - Git LFS (khuyên dùng):**
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add artifacts/cnn_mnist_model.keras
git commit -m "Add trained model with Git LFS"
git push
```

**Cách B - Render Disk:**
1. Trong Render Dashboard → Disks
2. Tạo disk 1GB, mount tại `/opt/render/project/src/artifacts`
3. Upload model qua SSH hoặc Render Shell

### Bước 6: Deploy
- Render tự động deploy khi push lên GitHub
- Hoặc click **"Manual Deploy"** → **"Deploy Latest Commit"**
- URL app: `https://mnist-digit-recognizer.onrender.com`

## 📊 Model Architecture

```
Input (28×28×1)
    ↓
Conv2D(32) + BN + ReLU   ← Học cạnh đơn giản
Conv2D(64) + BN + ReLU   ← Học đặc trưng phức tạp
MaxPool(2×2) + Dropout
    ↓
Conv2D(128) + BN + ReLU  ← Học hình dạng
Conv2D(128) + BN + ReLU  ← Học cấu trúc chữ số
MaxPool(2×2) + Dropout
    ↓
Flatten → Dense(256) + BN + ReLU + Dropout
    ↓
Dense(10) + Softmax      ← Output: P(0) ... P(9)
```

**Test Accuracy: >99%** trên 10,000 ảnh MNIST test set

## 📝 License
MIT License
# HeThongNhanDienChuSoVietTay
