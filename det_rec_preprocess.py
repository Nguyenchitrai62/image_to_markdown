import cv2
import numpy as np
from paddleocr import PaddleOCR
import paddle
import os
import json

# Import XY-Cut functions
from xycut import recursive_xy_cut, points_to_bbox

# Global OCR instance - khởi tạo một lần duy nhất
_ocr_instance = None

def get_ocr_instance():
    """Lấy OCR instance (singleton pattern)"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_recognition_model_dir="./PP-OCRv5_server_rec_infer",
            text_recognition_model_name="PP-OCRv5_server_rec",
            text_detection_model_name="PP-OCRv5_server_det",
        )
    return _ocr_instance

def initialize_ocr():
    """Khởi tạo PaddleOCR (deprecated - sử dụng get_ocr_instance() thay thế)"""
    return get_ocr_instance()

def convert_bbox_to_xycut_format(bbox):
    """
    Chuyển đổi bbox từ PaddleOCR format sang [x_min, y_min, x_max, y_max] cho XY-Cut
    
    Args:
        bbox: PaddleOCR bbox format - list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    Returns:
        list: [x_min, y_min, x_max, y_max]
    """
    try:
        if isinstance(bbox, list) and len(bbox) == 4:
            if isinstance(bbox[0], list):
                # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                result = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            else:
                # Format: [x1, y1, x2, y2] - giả sử đây là min-max format
                result = bbox
        elif hasattr(bbox, 'shape') and len(bbox.shape) == 2:
            # Numpy array shape (4, 2)
            x_coords = bbox[:, 0]
            y_coords = bbox[:, 1]
            result = [np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)]
        else:
            # Fallback - convert to numpy and try again
            bbox_array = np.array(bbox)
            if bbox_array.shape == (4, 2):
                x_coords = bbox_array[:, 0]
                y_coords = bbox_array[:, 1]
                result = [np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)]
            elif len(bbox_array) == 8:
                # Format: [x1,y1,x2,y2,x3,y3,x4,y4]
                result = points_to_bbox(bbox_array)
            else:
                result = [0, 0, 100, 100]  # Default fallback
        
        # Đảm bảo tất cả giá trị là integers và >= 0
        result = [max(0, int(x)) for x in result]
        return result
        
    except Exception:
        return [0, 0, 100, 100]  # Safe fallback

def sort_texts_with_xycut(texts_with_info):
    """
    Sắp xếp text theo thứ tự đọc sử dụng XY-Cut algorithm
    
    Args:
        texts_with_info: List of dicts với keys: 'text', 'score', 'bbox'
    
    Returns:
        List texts đã sắp xếp theo thứ tự đọc
    """
    if not texts_with_info:
        return []
    
    # Chuẩn bị dữ liệu cho XY-Cut
    boxes = []
    indices = list(range(len(texts_with_info)))
    
    for i, text_info in enumerate(texts_with_info):
        bbox = text_info.get('bbox', [[0, 0], [0, 0], [0, 0], [0, 0]])
        # Chuyển về format [x_min, y_min, x_max, y_max]
        converted_bbox = convert_bbox_to_xycut_format(bbox)
        boxes.append(converted_bbox)
    
    boxes = np.array(boxes, dtype=np.int32)
    indices = np.array(indices, dtype=np.int32)
    
    # Kiểm tra dữ liệu đầu vào
    if len(boxes) == 0:
        return []
    
    try:
        # Chạy XY-Cut để lấy thứ tự đọc
        reading_order = []
        recursive_xy_cut(boxes, indices, reading_order)
        
        # Sắp xếp texts theo thứ tự đọc
        if reading_order:
            sorted_texts = [texts_with_info[i] for i in reading_order]
        else:
            # Fallback về sắp xếp đơn giản nếu XY-Cut không trả về kết quả
            sorted_texts = sort_texts_by_position_fallback(texts_with_info)
            
    except Exception:
        # Fallback về sắp xếp đơn giản nếu XY-Cut thất bại
        sorted_texts = sort_texts_by_position_fallback(texts_with_info)
    
    return sorted_texts

def sort_texts_by_position_fallback(texts_with_info):
    """Sắp xếp text theo vị trí từ trên xuống dưới, trái sang phải (fallback method)"""
    def get_sort_key(text_info):
        bbox = text_info.get('bbox', [[0, 0], [0, 0], [0, 0], [0, 0]])
        try:
            # Tính tọa độ trung tâm
            if isinstance(bbox[0], list):
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
            else:
                # Fallback
                center_x, center_y = 0, 0
            return (center_y, center_x)
        except:
            return (0, 0)
    
    return sorted(texts_with_info, key=get_sort_key)

def get_text(image_input, save_output: bool = False, output_name: str = "output", sort_by_reading_order: bool = True) -> str:
    """
    Chạy OCR trên một ảnh với tính năng sắp xếp text theo thứ tự đọc sử dụng XY-Cut.

    Args:
        image_input (str or np.ndarray): Đường dẫn ảnh hoặc ảnh numpy array (BGR).
        save_output (bool): Nếu True, lưu ảnh kết quả và JSON.
        output_name (str): Tên cơ sở để lưu ảnh/JSON nếu cần.
        sort_by_reading_order (bool): Nếu True, sắp xếp text theo thứ tự đọc bằng XY-Cut.

    Returns:
        str: Văn bản đã nhận dạng từ ảnh, đã sắp xếp theo thứ tự đọc.
    """
    ocr = get_ocr_instance()

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

    all_texts_with_info = []

    for i, res in enumerate(result):
        angle = res.get("doc_preprocessor_res", {}).get("angle", None)
        
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("dt_polys", [])

        # Tạo list các text với thông tin đầy đủ
        for text, score, box in zip(texts, scores, boxes):
            text_info = {
                'text': text,
                'score': float(score),
                'bbox': box.tolist() if hasattr(box, 'tolist') else box
            }
            all_texts_with_info.append(text_info)

    # Sắp xếp text theo thứ tự đọc nếu được yêu cầu
    if sort_by_reading_order and all_texts_with_info:
        sorted_texts_info = sort_texts_with_xycut(all_texts_with_info)
        # Extract chỉ text đã được sắp xếp
        all_texts = [info['text'] for info in sorted_texts_info]
    else:
        # Không sắp xếp, giữ nguyên thứ tự ban đầu
        all_texts = [info['text'] for info in all_texts_with_info]

    return " ".join(all_texts)

def get_text_with_details(image_input, sort_by_reading_order: bool = True) -> dict:
    """
    Chạy OCR và trả về thông tin chi tiết bao gồm text, scores, bboxes đã sắp xếp theo XY-Cut.
    
    Args:
        image_input (str or np.ndarray): Đường dẫn ảnh hoặc ảnh numpy array (BGR).
        sort_by_reading_order (bool): Nếu True, sắp xếp text theo thứ tự đọc bằng XY-Cut.
    
    Returns:
        dict: {
            'texts': List[str],           # Danh sách text đã sắp xếp
            'text_infos': List[dict],     # Thông tin chi tiết từng text
            'combined_text': str          # Text được nối lại thành chuỗi
        }
    """
    ocr = get_ocr_instance()

    # Kiểm tra đầu vào
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

    all_texts_with_info = []

    for i, res in enumerate(result):
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("dt_polys", [])

        for text, score, box in zip(texts, scores, boxes):
            text_info = {
                'text': text,
                'confidence': float(score),
                'bbox': box.tolist() if hasattr(box, 'tolist') else box
            }
            all_texts_with_info.append(text_info)

    # Sắp xếp text theo thứ tự đọc
    if sort_by_reading_order and all_texts_with_info:
        sorted_texts_info = sort_texts_with_xycut(all_texts_with_info)
    else:
        sorted_texts_info = all_texts_with_info

    # Trả về kết quả
    texts = [info['text'] for info in sorted_texts_info]
    combined_text = " ".join(texts)

    return {
        'texts': texts,
        'text_infos': sorted_texts_info,
        'combined_text': combined_text
    }

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

def run_det_rec_preprocess(image_input, ocr_model=None):
    """
    Hàm chính: Xử lý ảnh với OCR và sắp xếp text theo XY-Cut
    
    Args:
        image_input (str or np.ndarray): Đường dẫn ảnh hoặc numpy array ảnh
        ocr_model: OCR model (deprecated - sẽ sử dụng global instance)
    
    Returns:
        tuple: (processed_image, recognized_texts, detection_boxes)
    """
    # Sử dụng global OCR instance thay vì parameter
    ocr = get_ocr_instance()
    
    # Kiểm tra đầu vào là path hay numpy array
    if isinstance(image_input, str):
        # Input là đường dẫn file
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Không thể tìm thấy file: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {image_input}")
        input_data = image_input
    elif isinstance(image_input, np.ndarray):
        # Input là numpy array
        img = image_input.copy()
        input_data = image_input
    else:
        raise TypeError("image_input phải là đường dẫn (str) hoặc ảnh numpy array (np.ndarray)")
    
    # Chạy OCR
    result = ocr.predict(input=input_data)
    
    all_texts_with_info = []
    all_boxes = []
    
    for res in result:
        texts = res.get("rec_texts", [])
        scores = res.get("rec_scores", [])
        boxes = res.get("dt_polys", [])
        
        # Tạo danh sách text với confidence scores
        for text, score, box in zip(texts, scores, boxes):
            text_info = {
                'text': text,
                'confidence': float(score),
                'bbox': box.tolist() if hasattr(box, 'tolist') else box
            }
            all_texts_with_info.append(text_info)
        
        all_boxes.extend(boxes)
        
        # Xử lý ảnh
        if len(boxes) > 0:
            text_mask = create_text_mask(img.shape, boxes)
            processed_image = preprocess_non_text_areas(img, text_mask)
        else:
            processed_image = img.copy()
    
    # Sắp xếp text bằng XY-Cut
    if all_texts_with_info:
        sorted_texts = sort_texts_with_xycut(all_texts_with_info)
    else:
        sorted_texts = all_texts_with_info
    
    return processed_image, sorted_texts, all_boxes

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
# 1. Test với hàm mới (OCR + XY-Cut sorting):
#    text = get_text("./anh_test/1.jpg", sort_by_reading_order=True)
#    print("Text nhận dạng (đã sắp xếp):", text)
#
# 2. Test với hàm chi tiết:
#    result = get_text_with_details("./anh_test/1.jpg")
#    print("Texts sorted:", result['texts'])
#    print("Combined:", result['combined_text'])
#
# 3. Test với numpy array:
#    img = cv2.imread("./anh_test/1.jpg")
#    cropped = img[100:300, 50:400]
#    text = get_text(cropped, sort_by_reading_order=True)
#    print("Text từ vùng cắt (đã sắp xếp):", text)
#
# 4. So sánh với/không XY-Cut:
#    text_sorted = get_text("./anh_test/1.jpg", sort_by_reading_order=True)
#    text_original = get_text("./anh_test/1.jpg", sort_by_reading_order=False)
#    print("Với XY-Cut:", text_sorted)
#    print("Không XY-Cut:", text_original)

if __name__ == "__main__":
    print("=== TEST HÀML MỚI: get_text với XY-Cut ===")
    image_path = "./anh_test/1.jpg"
    
    # Test với file ảnh
    if os.path.exists(image_path):
        print("\n1️⃣ Test OCR với XY-Cut sorting:")
        text_result = get_text(image_path, sort_by_reading_order=True)
        print(f"Text nhận dạng (sắp xếp XY-Cut): {text_result}")
        
        print("\n2️⃣ Test OCR không sorting:")
        text_no_sort = get_text(image_path, sort_by_reading_order=False)
        print(f"Text nhận dạng (không sắp xếp): {text_no_sort}")
        
        print("\n3️⃣ Test chi tiết với XY-Cut:")
        detailed_result = get_text_with_details(image_path)
        print(f"Số text regions: {len(detailed_result['text_infos'])}")
        print(f"Texts riêng lẻ: {detailed_result['texts']}")
        print(f"Combined text: {detailed_result['combined_text']}")
        
        # Test với numpy array (cắt ảnh)
        print("\n4️⃣ Test với numpy array (cropped image):")
        img = cv2.imread(image_path)
        if img is not None:
            cropped_img = img[100:300, 50:400]  # Cắt một vùng
            text_from_crop = get_text(cropped_img, sort_by_reading_order=True)
            print(f"Text từ vùng cắt (với XY-Cut): {text_from_crop}")
    else:
        print(f"File {image_path} không tồn tại. Tạo file test hoặc thay đổi đường dẫn.")
    
    print("\n=== TEST HÀM CŨ: run_det_rec_preprocess với XY-Cut ===")
    # Test hàm cũ để đảm bảo tương thích ngược
    if os.path.exists(image_path):
        processed_img, texts, boxes = run_det_rec_preprocess(image_path)
        print(f"Kích thước ảnh xử lý: {processed_img.shape}")
        print(f"Số text tìm được: {len(texts)}")
        for i, text in enumerate(texts[:5]):  # Hiển thị 5 text đầu
            print(f"Text {i+1}: '{text['text']}' (confidence: {text['confidence']:.3f})")
        
        # Lưu kết quả để test
        save_results(processed_img, texts, image_path)
        
        print(f"\n📝 Text được sắp xếp theo XY-Cut:")
        sorted_text_string = " ".join([t['text'] for t in texts])
        print(sorted_text_string)