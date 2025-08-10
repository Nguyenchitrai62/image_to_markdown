# Image to PDF (Image → Markdown → PDF)

Công cụ chuyển đổi ảnh thành văn bản (Markdown) và xuất ra PDF, sử dụng **PaddleOCR** để nhận dạng ký tự và bảng.

## 🚀 Cài đặt

### 1️⃣ Cài PaddlePaddle GPU

Nếu dùng CUDA 11.8:

```bash
python -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

Hoặc tham khảo hướng dẫn cài đặt nhanh tại [đây](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/windows-pip_en.html)

> 💡 Lưu ý: Nếu máy không có GPU hoặc CUDA không tương thích, bạn có thể cài `paddlepaddle` bản CPU (chậm hơn).

### 2️⃣ Cài thư viện Python cần thiết

```bash
pip install -r requirements.txt
```

## 📌 Cách sử dụng

Chạy script **img\_2\_md.py** để xử lý ảnh:

```bash
python img_2_md.py
```

Script sẽ:

1. Nhận diện văn bản và bảng từ ảnh.
2. Xuất kết quả sang Markdown.
3. Chuyển Markdown sang PDF.

## 📂 Cấu trúc dự án

```
├── img_2_md.py         # File chính để chạy
├── requirements.txt    # Danh sách thư viện cần cài
├── README.md           # Hướng dẫn
└── ...                 # Các file/thư mục khác
```

## 📜 Giấy phép

MIT License
