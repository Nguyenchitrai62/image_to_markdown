# Image → Markdown

## 1. Tổng quan

Công cụ cho phép **nhận dạng văn bản và bảng từ ảnh tài liệu tiếng Việt**, sau đó tự động chuyển đổi sang định dạng Markdown.
Hệ thống sử dụng **mô hình PaddleOCR đã được fine-tune chuyên biệt cho tiếng Việt**, giúp nhận diện chính xác dấu câu, ký tự đặc thù và cấu trúc phức tạp của tài liệu.

Bảng biểu được xử lý bằng **mô-đun nhận diện cell có sẵn của PaddleOCR**, sau đó sử dụng **tool tự phát triển** để sinh HTML từ bounding box của cell và nhúng trực tiếp vào file Markdown. Cách tiếp cận này giúp tái tạo chính xác cấu trúc bảng (hàng, cột, ô gộp) và hiển thị đúng layout trong trình đọc Markdown hỗ trợ HTML.

## 2. Điểm nổi bật

* **Fine-tune model tiếng Việt**: Mô hình recognition đã được huấn luyện lại với từ điển và dataset tiếng Việt.
* **Xử lý bảng nâng cao**: Kết hợp module Table Recognition của PaddleOCR và xử lý hậu kỳ để tái tạo bảng dưới dạng Markdown/HTML.
* **Pipeline hoàn chỉnh**: Ảnh đầu vào → nhận dạng layout → nhận dạng văn bản và bảng → xuất Markdown.
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

Nếu không dùng CUDA 11.8 hoặc sử dụng CPU, vui lòng truy cập **[trang cài đặt PaddlePaddle](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/windows-pip_en.html)** để chọn phiên bản phù hợp.

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
3. Sinh HTML bảng từ cell box và nhúng vào Markdown.
4. Xuất kết quả ra file `.md`.

## 5. Dataset

* **Nguồn dữ liệu**: Bộ dữ liệu tự thu thập từ nhiều nguồn public.
* **Quy mô**: Khoảng **450.000 ảnh public** và **150.000 ảnh sinh thêm** để cân bằng phân bố ký tự.
* **Mục tiêu**: Đảm bảo mô hình nhận dạng tốt các dạng chữ, dấu tiếng Việt.
* **Liên hệ**: Nếu cần dataset, vui lòng liên hệ email **[trainguyenchi30@gmail.com](mailto:trainguyenchi30@gmail.com)**.
