from paddleocr import TextRecognition
model = TextRecognition(model_name="PP-OCRv5_server_rec",model_dir="PP-OCRv5_server_rec_infer")
output = model.predict(input="./anh_test/6.jpg", batch_size=1)
for res in output:
    rec_texts = res["rec_texts"]
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")