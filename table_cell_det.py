import os
import time
import paddle
from paddleocr import TableCellsDetection

# Khởi tạo model
start_time_init = time.time()
model = TableCellsDetection(
    model_name="RT-DETR-L_wired_table_cell_det",
    # device='cpu'  # Chỉ định rõ ràng device là CPU
)
end_time_init = time.time()
print(f"Thời gian khởi tạo model: {end_time_init - start_time_init:.2f} giây")

# Dự đoán
start_time_pred = time.time()
output = model.predict("./anh_test/7.jpg", threshold=0.7, batch_size=1)
end_time_pred = time.time()
print(f"Thời gian dự đoán: {end_time_pred - start_time_pred:.2f} giây")

# Xử lý kết quả
start_time_post = time.time()
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    # res.save_to_json("./output/res.json")

for idx, res in enumerate(output):
    for i, box_info in enumerate(res["boxes"]):
        label = box_info["label"]
        coordinate = box_info["coordinate"]  # [x1, y1, x2, y2, x3, y3, x4, y4]
end_time_post = time.time()
print(f"Thời gian xử lý kết quả: {end_time_post - start_time_post:.2f} giây")

# Tổng thời gian
print(f"Tổng thời gian chạy: {end_time_post - start_time_init:.2f} giây")