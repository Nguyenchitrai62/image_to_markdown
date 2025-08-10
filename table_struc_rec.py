from paddleocr import TableStructureRecognition
import cv2
import numpy as np
import os

# Chuyển structure list -> HTML string
def structure_list_to_html(structure_list):
    html_lines = []
    buffer = ""
    for token in structure_list:
        if token.startswith("<") and ">" in token:
            if buffer:
                html_lines.append(buffer)
                buffer = ""
            if ">" in token and not token.endswith(">"):
                buffer = token
            else:
                html_lines.append(token)
        else:
            buffer += token
            if token.endswith(">"):
                html_lines.append(buffer)
                buffer = ""
    if buffer:
        html_lines.append(buffer)
    return "\n".join(html_lines)

# Vẽ bbox lên ảnh (dùng res["bbox"])
def draw_bboxes_on_image(image_path, bboxes, save_path="output/visualized.jpg"):
    image = cv2.imread(image_path)
    for i, box in enumerate(bboxes):
        pts = np.array(box, dtype=np.int32).reshape((4, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(image, f"bbox_{i}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)
    print(f"✅ Đã lưu ảnh minh họa: {save_path}")

# Load model SLANet
model = TableStructureRecognition(model_name="SLANet_plus")

# Ảnh đầu vào
image_path = "./output/after_processing.jpg"
output = model.predict(input=image_path, batch_size=1)

# Xử lý kết quả
for idx, res in enumerate(output):
    # 1. Chuyển structure sang HTML
    structure = res["structure"]
    html_content = structure_list_to_html(structure)

    # 2. Lưu HTML vào .md
    md_path = f"./output/table_{idx}.md"
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ Đã lưu Markdown: {md_path}")

    # 3. Vẽ bbox từ res["bbox"]
    bboxes = res["bbox"]  # Là danh sách các bbox 8 điểm
    draw_bboxes_on_image(res["input_path"], bboxes, save_path=f"./output/visualized_{idx}.jpg")
