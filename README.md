# Image → Markdown

## 1. Tổng quan

Công cụ này được phát triển để **nhận dạng văn bản và bảng từ ảnh tiếng Việt** với độ chính xác cao, sau đó chuyển đổi sang định dạng Markdown. Hệ thống sử dụng **mô hình PaddleOCR đã fine-tune riêng cho tiếng Việt**, đảm bảo khả năng nhận diện tốt dấu tiếng Việt, ký tự đặc thù và định dạng phức tạp của tài liệu.

## 2. Điểm nổi bật

* **Fine-tune model tiếng Việt**: Mô hình recognition trong repo là bản đã được huấn luyện lại với từ điển và dataset tiếng Việt.
* **Xử lý bảng nâng cao**: Kết hợp module Table Recognition của PaddleOCR và xử lý hậu kỳ để tái tạo bảng dưới dạng Markdown/HTML.
* **Pipeline hoàn chỉnh**: Từ ảnh đầu vào → nhận dạng layout → nhận dạng văn bản và bảng → xuất Markdown.
* **Hiệu năng tối ưu**: Hỗ trợ GPU qua PaddlePaddle-GPU.

## 3. Hướng dẫn cài đặt

### 3.1 Tạo và kích hoạt môi trường ảo bằng pip (khuyến nghị)

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Cập nhật pip lên bản mới nhất
pip install --upgrade pip
```

### 3.2 Cài PaddlePaddle GPU (khuyến nghị)

Nếu dùng CUDA 11.8:

```bash
pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

Nếu không dùng CUDA 11.8 hoặc sử dụng CPU, vui lòng truy cập tại **[đây](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/windows-pip_en.html)** để chọn phiên bản phù hợp.

### 3.3 Cài thư viện Python cần thiết

```bash
pip install -r requirements.txt
```

## 4. Hướng dẫn sử dụng

Chạy script **main.py**:

```bash
python main.py
```

Pipeline sẽ thực hiện:

1. Nhận diện layout tài liệu (heading, paragraph, table...).
2. Nhận dạng văn bản và bảng bằng mô hình tiếng Việt đã fine-tune.
3. Xuất kết quả ra file Markdown.

## 5. Dataset

* **Nguồn dữ liệu**: Bộ dữ liệu tự thu thập từ nhiều nguồn public.
* **Quy mô**: Khoảng **450.000 ảnh public** và **150.000 ảnh sinh thêm** để cân bằng phân bố ký tự.
* **Mục tiêu**: Đảm bảo mô hình nhận dạng tốt các dạng chữ, dấu tiếng Việt và cấu trúc bảng.
* **Liên hệ**: Nếu cần dataset, vui lòng email **[trainguyenchi30@gmail.com](mailto:trainguyenchi30@gmail.com)**.

