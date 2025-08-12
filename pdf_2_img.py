from pdf2image import convert_from_path
import os

pdf_path = "Z35.pdf"
output_dir = "pdf_pages"
os.makedirs(output_dir, exist_ok=True)

# Chỉ định poppler_path nếu cần
poppler_path = "C:/poppler/Library/bin"

# Lấy từng trang (generator) để tránh load toàn bộ vào RAM
for page_num, image in enumerate(
    convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path, fmt="jpeg"), start=1
):
    out_path = os.path.join(output_dir, f"page_{page_num}.jpg")
    image.save(out_path, "JPEG")
    print(f"Saved: {out_path}")
