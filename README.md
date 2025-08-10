# 📄 Image → Markdown

## 1️⃣ Tổng quan

**Image → Markdown** là công cụ giúp **nhận dạng văn bản và bảng từ ảnh tài liệu tiếng Việt** với độ chính xác cao, sau đó tự động chuyển đổi sang định dạng Markdown.
Nhờ sử dụng **mô hình PaddleOCR đã fine-tune chuyên biệt cho tiếng Việt**, hệ thống nhận diện tốt dấu câu, ký tự đặc thù và cả cấu trúc tài liệu phức tạp.

📊 **Bảng biểu** được phát hiện bằng **mô-đun cell detection của PaddleOCR**. Sau đó, công cụ **tự phát triển** sẽ:

* Chuyển bounding box cell → HTML `<table>`
* Giữ nguyên hàng, cột, ô gộp
* Nhúng HTML trực tiếp vào Markdown để hiển thị đúng layout.

---

## ✨ Điểm nổi bật

* 🚀 **Fine-tune model tiếng Việt** — Huấn luyện lại với từ điển & dataset tiếng Việt.
* 🗂 **Xử lý bảng nâng cao** — Kết hợp Table Recognition + hậu xử lý tái tạo bảng chuẩn HTML/Markdown.
* 🔄 **Pipeline hoàn chỉnh** — Ảnh đầu vào → layout detection → OCR → xuất Markdown.
* ⚡ **Hiệu năng tối ưu** — Hỗ trợ GPU qua PaddlePaddle-GPU.

---

## 🛠 Hướng dẫn cài đặt

### 3.1 Tạo & kích hoạt môi trường ảo (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install --upgrade pip
```

### 3.2 Cài PaddlePaddle GPU (nếu dùng CUDA 11.8)

```bash
pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

Hoặc [chọn phiên bản khác tại đây](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/windows-pip_en.html).

### 3.3 Cài thư viện Python cần thiết

```bash
pip install -r requirements.txt
```

---

## ▶ Hướng dẫn sử dụng

```bash
python main.py
```

Pipeline sẽ tự động:

1. 📐 Nhận diện layout (heading, paragraph, table...)
2. 🔍 Nhận dạng văn bản & bảng bằng model tiếng Việt
3. 🧩 Sinh HTML bảng từ cell boxes → nhúng vào Markdown
4. 💾 Xuất kết quả `.md`

---

## 📚 Dataset

* **Nguồn**: Tự thu thập từ nhiều nguồn public.
* **Quy mô**: \~450.000 ảnh public + 150.000 ảnh sinh thêm.
* **Mục tiêu**: Tối ưu nhận dạng tiếng Việt với nhiều loại ký tự & dấu câu.
* **Liên hệ**: [trainguyenchi30@gmail.com](mailto:trainguyenchi30@gmail.com)
