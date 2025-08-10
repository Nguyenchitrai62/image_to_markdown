# Initialize PaddleOCR instance
from paddleocr import PaddleOCR
import paddle
print("Paddle đang dùng thiết bị:", paddle.device.get_device())

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    # lang='vi',
    text_recognition_model_dir ="./PP-OCRv5_server_rec_infer",
    text_recognition_model_name = "PP-OCRv5_server_rec",
    text_detection_model_name = "PP-OCRv5_server_det",
    )
 
# Run OCR inference on a sample image 
result = ocr.predict(input="./anh_test/1.jpg")

for res in result:
    angle = res.get("doc_preprocessor_res", {}).get("angle", None)
    if angle is not None:
        print(f"[Hướng ảnh] Angle: {angle}°")


    # texts = res["rec_texts"]       # Danh sách các đoạn văn bản đã nhận dạng
    # scores = res["rec_scores"]     # Danh sách độ chính xác tương ứng
    # boxes = res["dt_polys"]       # Danh sách tọa độ khung của từng văn bản (4 điểm)

    # for text, score, box in zip(texts, scores, boxes):
    #     print(f"Text: {text}, Score: {score:.2f}, Box: {box}")

# # Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
    
# python tools/export_model.py -c rec_mv3_none_bilstm_ctc.yml -o Global.pretrained_model=output/PP-OCRv5_server_rec/best_accuracy.pdparams Global.save_inference_dir="./PP-OCRv5_server_rec_infer/"
    