from paddleocr import TableCellsDetection
model = TableCellsDetection(model_name="RT-DETR-L_wired_table_cell_det")
output = model.predict("./cropped_boxes/table_0.jpg", threshold=0.7, batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    # res.save_to_json("./output/res.json")
    
for idx, res in enumerate(output):
    for i, box_info in enumerate(res["boxes"]):
        label = box_info["label"]
        coordinate = box_info["coordinate"]  # [x1, y1, x2, y2, x3, y3, x4, y4]