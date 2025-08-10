from paddleocr import PaddleOCR
import paddle
import numpy as np
import cv2
import os

# Khởi tạo PaddleOCR instance một lần (chỉ nên gọi một lần trong toàn bộ pipeline)
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_recognition_model_dir="./PP-OCRv5_server_rec_infer",
    text_recognition_model_name="PP-OCRv5_server_rec",
    text_detection_model_name="PP-OCRv5_server_det",
)

def run_ocr_on_image_input(image_input, save_output: bool = False, output_name: str = "output") -> str:
    """
    Chạy OCR trên một ảnh (có thể là đường dẫn hoặc numpy array).

    Args:
        image_input (str or np.ndarray): Đường dẫn ảnh hoặc ảnh numpy array (BGR).
        save_output (bool): Nếu True, lưu ảnh kết quả và JSON.
        output_name (str): Tên cơ sở để lưu ảnh/JSON nếu cần.

    Returns:
        str: Văn bản đã nhận dạng từ ảnh.
    """
    print("Paddle đang dùng thiết bị:", paddle.device.get_device())

    # Kiểm tra đầu vào là path hay array
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Ảnh không tồn tại: {image_input}")
        input_data = image_input
    elif isinstance(image_input, np.ndarray):
        input_data = image_input
    else:
        raise TypeError("image_input phải là đường dẫn (str) hoặc ảnh numpy array (np.ndarray)")

    # Gọi OCR
    result = ocr.predict(input=input_data)

    all_texts = []

    for i, res in enumerate(result):
        angle = res.get("doc_preprocessor_res", {}).get("angle", None)
        # if angle is not None:
        #     print(f"[Hướng ảnh] Angle: {angle}°")

        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("dt_polys", [])

        for text, score, box in zip(texts, scores, boxes):
            all_texts.append(text)
            # print(f"Text: {text}, Score: {score:.2f}, Box: {box}")

        # if save_output:
        #     res.print()
        #     res.save_to_img(output_name + f"_{i}")
        #     res.save_to_json(output_name + f"_{i}")

    return " ".join(all_texts)

if __name__ == "__main__":
    print("=== TEST OCR VỚI FILE ẢNH ===")
    image_path = "./cropped_boxes/doc_title_3.jpg"
    text_from_file = run_ocr_on_image_input(image_path, save_output=True, output_name="result_from_file")
    print("\n📄 Kết quả từ ảnh file:")
    print(text_from_file)

    print("\n=== TEST OCR VỚI ẢNH CẮT (numpy array) ===")
    img = cv2.imread(image_path)

    # Cắt một vùng nhỏ trong ảnh (giả định vùng có text)
    cropped_img = img[100:200, 150:400]

    text_from_crop = run_ocr_on_image_input(cropped_img, save_output=True, output_name="result_from_crop")
    print("\n📄 Kết quả từ ảnh cắt:")
    print(text_from_crop)