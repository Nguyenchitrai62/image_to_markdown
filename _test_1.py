import cv2
import numpy as np
from paddleocr import PaddleOCR
import json
import os
from typing import List, Dict, Tuple, Any
from googletrans import Translator
import time

# Import XY-Cut functions (giả sử đã có)
from xycut import recursive_xy_cut, points_to_bbox

def calculate_iou(box1, box2):
    """
    Tính IoU (Intersection over Union) giữa hai bounding box
    
    Args:
        box1, box2: [x_min, y_min, x_max, y_max] format
        
    Returns:
        float: IoU value (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_overlap_ratio(box1, box2):
    """
    Tính tỷ lệ overlap của box nhỏ hơn so với box lớn hơn
    
    Args:
        box1, box2: [x_min, y_min, x_max, y_max] format
        
    Returns:
        float: Overlap ratio (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Tỷ lệ overlap của box nhỏ hơn
    min_area = min(area1, area2)
    return intersection / min_area if min_area > 0 else 0.0

def has_intersection(box1, box2):
    """
    Kiểm tra hai bounding box có giao nhau không (dù chỉ 1 pixel)
    
    Args:
        box1, box2: [x_min, y_min, x_max, y_max] format
        
    Returns:
        bool: True nếu có giao nhau, False nếu không
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Có giao nhau nếu x1 < x2 và y1 < y2
    return x1 < x2 and y1 < y2

def convert_bbox_to_xycut_format(bbox):
    """
    Chuyển đổi bbox từ PaddleOCR format sang [x_min, y_min, x_max, y_max]
    """
    try:
        if isinstance(bbox, list) and len(bbox) == 4:
            if isinstance(bbox[0], list):
                # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                result = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            else:
                # Format: [x1, y1, x2, y2]
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
                result = [0, 0, 100, 100]
        
        # Đảm bảo tất cả giá trị là integers và >= 0
        result = [max(0, int(x)) for x in result]
        return result
        
    except Exception:
        return [0, 0, 100, 100]

def find_overlapping_groups(text_infos):
    """
    Tìm các nhóm text boxes có chồng lên nhau (bất kỳ mức độ nào)
    
    Args:
        text_infos: List of dict với keys: 'text', 'confidence', 'bbox'
        
    Returns:
        List of lists: Mỗi sublist chứa indices của các box chồng lên nhau
    """
    n = len(text_infos)
    if n == 0:
        return []
    
    # Convert tất cả bbox về định dạng chuẩn
    boxes = []
    for info in text_infos:
        bbox = convert_bbox_to_xycut_format(info['bbox'])
        boxes.append(bbox)
    
    # Tạo ma trận adjacency
    overlap_matrix = np.zeros((n, n), dtype=bool)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Chỉ cần kiểm tra có giao nhau không (không cần ngưỡng)
            if has_intersection(boxes[i], boxes[j]):
                overlap_matrix[i][j] = True
                overlap_matrix[j][i] = True
    
    # Tìm các connected components (nhóm các box chồng lên nhau)
    visited = [False] * n
    groups = []
    
    def dfs(node, current_group):
        visited[node] = True
        current_group.append(node)
        
        for neighbor in range(n):
            if overlap_matrix[node][neighbor] and not visited[neighbor]:
                dfs(neighbor, current_group)
    
    for i in range(n):
        if not visited[i]:
            current_group = []
            dfs(i, current_group)
            groups.append(current_group)
    
    return groups

def merge_overlapping_boxes(text_infos, groups):
    """
    Gộp các box overlap thành box lớn và sắp xếp text bằng XY-Cut
    
    Args:
        text_infos: List of text info dicts
        groups: List of lists, mỗi sublist chứa indices của các box cần gộp
        
    Returns:
        List of merged text info dicts
    """
    merged_results = []
    
    for group in groups:
        if len(group) == 1:
            # Chỉ có 1 box, không cần gộp
            merged_results.append(text_infos[group[0]])
        else:
            # Gộp nhiều box
            group_texts = []
            group_confidences = []
            all_bboxes = []
            
            # Collect thông tin của tất cả box trong group
            for idx in group:
                info = text_infos[idx]
                group_texts.append({
                    'text': info['text'],
                    'confidence': info['confidence'],
                    'bbox': info['bbox']
                })
                group_confidences.append(info['confidence'])
                all_bboxes.append(convert_bbox_to_xycut_format(info['bbox']))
            
            # Tính bounding box gộp
            all_bboxes = np.array(all_bboxes)
            merged_bbox = [
                int(np.min(all_bboxes[:, 0])),  # x_min
                int(np.min(all_bboxes[:, 1])),  # y_min
                int(np.max(all_bboxes[:, 2])),  # x_max
                int(np.max(all_bboxes[:, 3]))   # y_max
            ]
            
            # Sắp xếp text trong group bằng XY-Cut
            sorted_group_texts = sort_texts_with_xycut(group_texts)
            
            # Gộp text và tính confidence trung bình
            combined_text = " ".join([t['text'] for t in sorted_group_texts])
            avg_confidence = np.mean(group_confidences)
            
            merged_info = {
                'text': combined_text,
                'confidence': float(avg_confidence),
                'bbox': merged_bbox,
                'is_merged': True,
                'merged_from_count': len(group),
                'original_texts': [t['text'] for t in sorted_group_texts]
            }
            
            merged_results.append(merged_info)
    
    return merged_results

def sort_texts_with_xycut(texts_with_info):
    """
    Sắp xếp text theo thứ tự đọc sử dụng XY-Cut algorithm
    """
    if not texts_with_info:
        return []
    
    # Chuẩn bị dữ liệu cho XY-Cut
    boxes = []
    indices = list(range(len(texts_with_info)))
    
    for i, text_info in enumerate(texts_with_info):
        bbox = text_info.get('bbox', [[0, 0], [0, 0], [0, 0], [0, 0]])
        converted_bbox = convert_bbox_to_xycut_format(bbox)
        boxes.append(converted_bbox)
    
    boxes = np.array(boxes, dtype=np.int32)
    indices = np.array(indices, dtype=np.int32)
    
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
            # Fallback về sắp xếp đơn giản
            sorted_texts = sort_texts_by_position_fallback(texts_with_info)
            
    except Exception:
        # Fallback về sắp xếp đơn giản
        sorted_texts = sort_texts_by_position_fallback(texts_with_info)
    
    return sorted_texts

def sort_texts_by_position_fallback(texts_with_info):
    """Fallback sorting method"""
    def get_sort_key(text_info):
        bbox = text_info.get('bbox', [[0, 0], [0, 0], [0, 0], [0, 0]])
        try:
            if isinstance(bbox[0], list):
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
            else:
                center_x, center_y = bbox[0], bbox[1]
            return (center_y, center_x)
        except:
            return (0, 0)
    
    return sorted(texts_with_info, key=get_sort_key)

def translate_text(text, src_lang="ja", dest_lang="vi", max_retries=3):
    """
    Dịch text sử dụng Google Translate với retry mechanism
    
    Args:
        text: Text cần dịch
        src_lang: Ngôn ngữ nguồn
        dest_lang: Ngôn ngữ đích
        max_retries: Số lần thử lại tối đa
        
    Returns:
        str: Text đã dịch
    """
    if not text or not text.strip():
        return text
    
    translator = Translator()
    
    for attempt in range(max_retries):
        try:
            result = translator.translate(text, src=src_lang, dest=dest_lang)
            return result.text
        except Exception as e:
            print(f"⚠️ Lỗi dịch lần {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Đợi 1 giây trước khi thử lại
            else:
                print(f"❌ Không thể dịch text: '{text}', giữ nguyên text gốc")
                return text
    
    return text

def process_image_to_json(image_input, 
                         src_lang="ja",
                         dest_lang="vi",
                         output_json_path=None,
                         save_debug_image=False):
    """
    Main function: Xử lý ảnh OCR, gộp box, dịch text và xuất JSON
    
    Args:
        image_input: Đường dẫn ảnh hoặc numpy array
        src_lang: Ngôn ngữ nguồn (mặc định "ja")
        dest_lang: Ngôn ngữ đích (mặc định "vi")
        output_json_path: Đường dẫn lưu JSON
        save_debug_image: Có lưu ảnh debug không
        
    Returns:
        dict: Kết quả xử lý
    """
    # Khởi tạo OCR
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_model_name="PP-OCRv5_server_rec",
        text_detection_model_name="PP-OCRv5_server_det",
    )
    
    # Kiểm tra input
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Ảnh không tồn tại: {image_input}")
        input_data = image_input
        base_name = os.path.splitext(os.path.basename(image_input))[0]
        image_path = image_input
    elif isinstance(image_input, np.ndarray):
        input_data = image_input
        base_name = "image_array"
        image_path = None
    else:
        raise TypeError("image_input phải là đường dẫn hoặc numpy array")
    
    # Tạo thư mục output
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Chạy OCR
    print("🔍 Đang chạy OCR...")
    result = ocr.predict(input=input_data)
    
    # Extract text information
    all_texts_with_info = []
    for res in result:
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
    
    print(f"📊 Tìm thấy {len(all_texts_with_info)} text boxes ban đầu")
    
    # Tìm và gộp các box chồng lên nhau
    print("🔄 Đang gộp các box chồng lên nhau...")
    overlapping_groups = find_overlapping_groups(all_texts_with_info)
    merged_texts = merge_overlapping_boxes(all_texts_with_info, overlapping_groups)
    
    # Sắp xếp toàn bộ kết quả theo XY-Cut
    print("📝 Đang sắp xếp text theo thứ tự đọc...")
    final_sorted_texts = sort_texts_with_xycut(merged_texts)
    
    # Dịch text
    print(f"🌍 Đang dịch text từ {src_lang} sang {dest_lang}...")
    for i, text_info in enumerate(final_sorted_texts):
        original_text = text_info['text']
        print(f"Đang dịch {i+1}/{len(final_sorted_texts)}: {original_text[:50]}...")
        
        translated_text = translate_text(original_text, src_lang, dest_lang)
        text_info['translated_text'] = translated_text
        text_info['original_text'] = original_text
        text_info['src_lang'] = src_lang
        text_info['dest_lang'] = dest_lang
        
        # time.sleep(0.1)  # Tránh spam API
    
    print("✅ Hoàn thành dịch text")
    
    # Thống kê
    merged_count = sum(1 for group in overlapping_groups if len(group) > 1)
    total_merged_boxes = sum(len(group) for group in overlapping_groups if len(group) > 1)
    
    print(f"📈 Kết quả gộp box:")
    print(f"- Box ban đầu: {len(all_texts_with_info)}")
    print(f"- Nhóm gộp: {merged_count}")
    print(f"- Box cuối cùng: {len(final_sorted_texts)}")
    
    # Chuẩn bị output data
    output_data = {
        'metadata': {
            'image_path': image_path,
            'total_original_boxes': len(all_texts_with_info),
            'total_merged_groups': merged_count,
            'total_final_boxes': len(final_sorted_texts),
            'source_language': src_lang,
            'target_language': dest_lang,
            'processing_timestamp': __import__('datetime').datetime.now().isoformat()
        },
        'texts': []
    }
    
    # Add text data
    for i, text_info in enumerate(final_sorted_texts):
        text_data = {
            'index': i,
            'original_text': text_info['original_text'],
            'translated_text': text_info['translated_text'],
            'confidence': text_info['confidence'],
            'bbox': text_info['bbox'],
            'is_merged': text_info.get('is_merged', False)
        }
        
        if text_info.get('is_merged', False):
            text_data['merged_from_count'] = text_info['merged_from_count']
            text_data['original_texts'] = text_info['original_texts']
        
        output_data['texts'].append(text_data)
    
    # Combined text (cả gốc và dịch)
    output_data['combined_original_text'] = " ".join([t['original_text'] for t in final_sorted_texts])
    output_data['combined_translated_text'] = " ".join([t['translated_text'] for t in final_sorted_texts])
    
    # Lưu JSON
    if output_json_path is None:
        output_json_path = os.path.join(output_dir, f"{base_name}_translated_ocr_result.json")
    else:
        output_json_path = os.path.join(output_dir, output_json_path)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Đã lưu kết quả JSON vào: {output_json_path}")
    
    # Lưu ảnh debug nếu cần
    if save_debug_image and image_path:
        debug_image_path = os.path.join(output_dir, f"{base_name}_debug_boxes.jpg")
        save_debug_visualization(image_path, final_sorted_texts, debug_image_path)
        print(f"🖼️ Đã lưu ảnh debug vào: {debug_image_path}")
    
    return output_data

def save_debug_visualization(image_path, text_infos, output_path):
    """Lưu ảnh visualization để debug"""
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # Vẽ bounding boxes
    for i, text_info in enumerate(text_infos):
        bbox = text_info['bbox']
        
        # Convert bbox to rectangle points
        if len(bbox) == 4 and not isinstance(bbox[0], list):
            # Format [x_min, y_min, x_max, y_max]
            x1, y1, x2, y2 = bbox
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        else:
            # Other formats
            converted = convert_bbox_to_xycut_format(bbox)
            x1, y1, x2, y2 = converted
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
        
        # Chọn màu: đỏ cho merged box, xanh cho single box
        color = (0, 0, 255) if text_info.get('is_merged', False) else (0, 255, 0)
        thickness = 3 if text_info.get('is_merged', False) else 2
        
        cv2.polylines(img, [pts], True, color, thickness)
        
        # Vẽ số thứ tự
        cv2.putText(img, str(i), (int(x1), int(y1)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imwrite(output_path, img)

# Test function
def test_ocr_to_json():
    """Test function chỉ xuất JSON"""
    image_path = "./cropped_boxes/Chap189-Page011.jpg"  # Thay đổi path này
    
    if os.path.exists(image_path):
        print("=== TEST OCR TO JSON ===")
        
        # Test dịch từ tiếng Nhật sang tiếng Việt
        print("\n🇯🇵➡️🇻🇳 Xử lý ảnh: Nhật → Việt")
        result = process_image_to_json(
            image_path,
            src_lang="ja",
            dest_lang="vi", 
            output_json_path="translated_ja_to_vi.json",
            save_debug_image=True
        )
        
        print(f"\n📈 Kết quả:")
        print(f"- Box ban đầu: {result['metadata']['total_original_boxes']}")
        print(f"- Nhóm gộp: {result['metadata']['total_merged_groups']}")
        print(f"- Box cuối cùng: {result['metadata']['total_final_boxes']}")
        
        print(f"\n📝 Text gốc (Nhật):")
        original_text = result['combined_original_text'][:200] + "..." if len(result['combined_original_text']) > 200 else result['combined_original_text']
        print(original_text)
        
        print(f"\n📝 Text dịch (Việt):")
        translated_text = result['combined_translated_text'][:200] + "..." if len(result['combined_translated_text']) > 200 else result['combined_translated_text']
        print(translated_text)
        
        print(f"\n💾 File JSON output: ./output/translated_ja_to_vi.json")
        
        # In chi tiết vài text đầu tiên
        print(f"\n🔍 Chi tiết 3 text đầu tiên:")
        for i, text_data in enumerate(result['texts'][:3]):
            print(f"{i+1}. Gốc: '{text_data['original_text']}'")
            print(f"   Dịch: '{text_data['translated_text']}'")
            print(f"   Confidence: {text_data['confidence']:.3f}")
            print(f"   Merged: {text_data['is_merged']}")
            print()
        
        return result
        
    else:
        print(f"❌ File {image_path} không tồn tại")
        print("Tạo thư mục cropped_boxes/ và đặt ảnh test vào đó")
        return None

def quick_process_to_json(image_path, src_lang="ja", dest_lang="vi"):
    """
    Hàm nhanh để xử lý ảnh ra JSON với tham số tối thiểu
    
    Args:
        image_path: Đường dẫn ảnh
        src_lang: Ngôn ngữ nguồn
        dest_lang: Ngôn ngữ đích
        
    Returns:
        str: Đường dẫn file JSON
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Ảnh không tồn tại: {image_path}")
    
    print(f"🚀 Xử lý nhanh: {src_lang} → {dest_lang}")
    
    result = process_image_to_json(
        image_path,
        src_lang=src_lang,
        dest_lang=dest_lang,
        save_debug_image=False
    )
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    json_path = f"./output/{base_name}_translated_ocr_result.json"
    
    print(f"✅ Hoàn thành! File JSON: {json_path}")
    return json_path

if __name__ == "__main__":
    # Chạy test cơ bản
    test_ocr_to_json()