import cv2
import numpy as np
from paddleocr import PaddleOCR
import paddle
import os
import json

def initialize_ocr():
    """Khởi tạo PaddleOCR"""
    print("Paddle đang dùng thiết bị:", paddle.device.get_device())
    
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_model_dir="./PP-OCRv5_server_rec_infer",
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_detection_model_name="PP-OCRv5_server_det",
    )
    
    return ocr

def create_text_mask(image_shape, dt_polys):
    """Tạo mask từ các bounding box text detection"""
    mask_uint8 = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    
    for poly in dt_polys:
        poly_int = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask_uint8, [poly_int], 255)
    
    mask = mask_uint8 > 0
    return mask

def preprocess_non_text_areas(img, text_mask):
    """Áp dụng tiền xử lý chỉ trên vùng không có text"""
    b, g, r = cv2.split(img)
    color_diff = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    gray_mask = (color_diff < 30) & (r < 200) & (g < 200) & (b < 200)
    preprocessing_mask = gray_mask & (~text_mask)
    
    output = np.ones_like(img) * 255
    output[text_mask] = img[text_mask]
    output[preprocessing_mask] = img[preprocessing_mask]
    
    return output

def run_det_rec_preprocess(image_path, ocr_model=None):
    """
    Hàm chính: Xử lý ảnh với OCR
    
    Returns:
        tuple: (processed_image, recognized_texts, detection_boxes)
    """
    if ocr_model is None:
        ocr_model = initialize_ocr()
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    print(f"Đang xử lý ảnh: {image_path}")
    
    # Chạy OCR
    result = ocr_model.predict(input=image_path)
    
    all_texts = []
    all_boxes = []
    
    for res in result:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("dt_polys", [])
        
        print(f"Tìm thấy {len(boxes)} vùng text")
        
        # Tạo danh sách text với confidence scores
        for text, score, box in zip(texts, scores, boxes):
            text_info = {
                'text': text,
                'confidence': float(score),
                'bbox': box.tolist() if hasattr(box, 'tolist') else box
            }
            all_texts.append(text_info)
        
        all_boxes.extend(boxes)
        
        # Xử lý ảnh
        if len(boxes) > 0:
            text_mask = create_text_mask(img.shape, boxes)
            processed_image = preprocess_non_text_areas(img, text_mask)
        else:
            processed_image = img.copy()
    
    return processed_image, all_texts, all_boxes

def save_results(processed_image, texts, image_path, output_dir="./output/"):
    """Lưu kết quả (chỉ để test)"""
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Lưu ảnh
    processed_path = os.path.join(output_dir, f"{base_name}_processed.jpg")
    cv2.imwrite(processed_path, processed_image)
    
    # Lưu text
    ocr_result_path = os.path.join(output_dir, f"{base_name}_ocr_result.json")
    with open(ocr_result_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    
    print(f"Đã lưu: {processed_path}")
    print(f"Đã lưu: {ocr_result_path}")

# CÁCH TEST:
# 1. Test cơ bản:
#    ocr_model = initialize_ocr()
#    processed_img, texts, boxes = run_det_rec_preprocess("./anh_test/1.jpg", ocr_model)
#    save_results(processed_img, texts, "./anh_test/1.jpg")
#
# 2. Test nhiều ảnh:
#    ocr_model = initialize_ocr()
#    for img_path in ["./anh_test/1.jpg", "./anh_test/2.jpg"]:
#        processed_img, texts, boxes = run_det_rec_preprocess(img_path, ocr_model)
#        save_results(processed_img, texts, img_path)

if __name__ == "__main__":
    # Test đơn giản
    ocr_model = initialize_ocr()
    image_path = "./anh_test/1.jpg"
    
    # Xử lý ảnh
    processed_img, texts, boxes = run_det_rec_preprocess(image_path, ocr_model)
    
    # In kết quả
    print(f"Kích thước ảnh: {processed_img.shape}")
    print(f"Số text tìm được: {len(texts)}")
    for i, text in enumerate(texts[:3]):
        print(f"Text {i+1}: '{text['text']}' ({text['confidence']:.3f})")
    
    # Lưu kết quả để test
    save_results(processed_img, texts, image_path)