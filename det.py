from paddleocr import TextDetection
model = TextDetection(model_name="PP-OCRv5_server_det")
output = model.predict("./cropped_boxes/table_0.jpg", batch_size=1)
for res in output:
    boxes = res["dt_polys"]
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")