import cv2
import numpy as np
from paddleocr import PaddleOCR
import json
import os
from typing import List, Dict, Tuple, Any
from PIL import Image, ImageDraw, ImageFont
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

def get_optimal_font_size(text, bbox_width, bbox_height, font_path, min_size=12, max_size=60):
    """
    Tính toán font size tối ưu để text vừa với bounding box
    
    Args:
        text: Text cần vẽ
        bbox_width: Chiều rộng bbox
        bbox_height: Chiều cao bbox
        font_path: Đường dẫn font
        min_size: Font size tối thiểu
        max_size: Font size tối đa
        
    Returns:
        int: Font size tối ưu
    """
    if not text:
        return min_size
    
    # Binary search để tìm font size tối ưu
    left, right = min_size, max_size
    optimal_size = min_size
    
    while left <= right:
        mid_size = (left + right) // 2
        try:
            font = ImageFont.truetype(font_path, mid_size)
            
            # Tính kích thước text với font size hiện tại
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Kiểm tra text có vừa trong bbox không (với margin 10%)
            margin_width = bbox_width * 0.1
            margin_height = bbox_height * 0.1
            
            if (text_width <= bbox_width - margin_width and 
                text_height <= bbox_height - margin_height):
                optimal_size = mid_size
                left = mid_size + 1
            else:
                right = mid_size - 1
                
        except Exception:
            right = mid_size - 1
    
    return max(optimal_size, min_size)

def draw_text_on_image(image, text_infos_with_translation, font_path="C:/Windows/Fonts/times.ttf"):
    """
    Vẽ text đã dịch lên ảnh với nền trắng
    
    Args:
        image: PIL Image object
        text_infos_with_translation: List các text info đã có translation
        font_path: Đường dẫn đến font
        
    Returns:
        PIL Image: Ảnh đã vẽ text
    """
    img_with_text = image.copy()
    draw = ImageDraw.Draw(img_with_text)
    
    # Kiểm tra font có tồn tại không
    if not os.path.exists(font_path):
        print(f"⚠️ Font không tồn tại: {font_path}")
        print("Sử dụng font mặc định")
        font_path = None
    
    for i, text_info in enumerate(text_infos_with_translation):
        try:
            bbox = text_info['bbox']
            translated_text = text_info.get('translated_text', text_info['text'])
            
            # Convert bbox
            if len(bbox) == 4 and not isinstance(bbox[0], list):
                x_min, y_min, x_max, y_max = bbox
            else:
                converted = convert_bbox_to_xycut_format(bbox)
                x_min, y_min, x_max, y_max = converted
            
            # Tính kích thước bbox
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            if bbox_width <= 0 or bbox_height <= 0:
                continue
            
            # Vẽ nền trắng
            draw.rectangle([x_min, y_min, x_max, y_max], fill='white', outline=None)
            
            # Tính font size tối ưu
            if font_path and os.path.exists(font_path):
                optimal_font_size = get_optimal_font_size(
                    translated_text, bbox_width, bbox_height, font_path
                )
                font = ImageFont.truetype(font_path, optimal_font_size)
            else:
                # Sử dụng font mặc định nếu không tìm thấy font
                try:
                    font = ImageFont.load_default()
                except:
                    continue
            
            # Tính vị trí để center text
            text_bbox = draw.textbbox((0, 0), translated_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Center text trong bbox
            text_x = x_min + (bbox_width - text_width) // 2
            text_y = y_min + (bbox_height - text_height) // 2
            
            # Đảm bảo text không vượt ra ngoài bbox
            text_x = max(x_min, min(text_x, x_max - text_width))
            text_y = max(y_min, min(text_y, y_max - text_height))
            
            # Vẽ text
            draw.text((text_x, text_y), translated_text, fill='black', font=font)
            
            # Vẽ viền để debug (tùy chọn)
            if text_info.get('is_merged', False):
                draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)
            else:
                draw.rectangle([x_min, y_min, x_max, y_max], outline='blue', width=1)
                
        except Exception as e:
            print(f"⚠️ Lỗi vẽ text {i}: {str(e)}")
            continue
    
    return img_with_text

def process_image_with_translation(image_input, 
                                   src_lang="ja",
                                   dest_lang="vi",
                                   output_json_path=None,
                                   save_debug_image=False,
                                   save_translated_image=True,
                                   font_path="C:/Windows/Fonts/times.ttf"):
    """
    Main function: Xử lý ảnh OCR, gộp box, dịch text và vẽ lên ảnh
    
    Args:
        image_input: Đường dẫn ảnh hoặc numpy array
        src_lang: Ngôn ngữ nguồn (mặc định "ja")
        dest_lang: Ngôn ngữ đích (mặc định "vi")
        output_json_path: Đường dẫn lưu JSON
        save_debug_image: Có lưu ảnh debug không
        save_translated_image: Có lưu ảnh đã dịch không
        font_path: Đường dẫn font
        
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
        # Load ảnh cho việc vẽ
        original_image = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        input_data = image_input
        base_name = "image_array"
        # Convert numpy array sang PIL Image
        if len(image_input.shape) == 3:
            original_image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            original_image = Image.fromarray(image_input)
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
    
    # Vẽ text dịch lên ảnh
    if save_translated_image:
        print("🎨 Đang vẽ text dịch lên ảnh...")
        translated_image = draw_text_on_image(original_image, final_sorted_texts, font_path)
        
        # Lưu ảnh đã dịch
        translated_image_path = os.path.join(output_dir, f"{base_name}_translated.jpg")
        translated_image.save(translated_image_path, quality=95)
        print(f"🖼️ Đã lưu ảnh dịch vào: {translated_image_path}")
    
    # Lưu ảnh debug nếu cần
    if save_debug_image and isinstance(image_input, str):
        debug_image_path = os.path.join(output_dir, f"{base_name}_debug_boxes.jpg")
        save_debug_visualization(image_input, final_sorted_texts, debug_image_path)
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
def test_ocr_translation():
    """Test function với dịch thuật"""
    image_path = "./cropped_boxes/Chap189-Page011.jpg"  # Thay đổi path này
    
    if os.path.exists(image_path):
        print("=== TEST OCR + TRANSLATION + TEXT OVERLAY ===")
        
        # Test dịch từ tiếng Nhật sang tiếng Việt
        print("\n🇯🇵➡️🇻🇳 Xử lý ảnh: Nhật → Việt")
        result = process_image_with_translation(
            image_path,
            src_lang="ja",
            dest_lang="vi", 
            output_json_path="translated_ja_to_vi.json",
            save_debug_image=True,
            save_translated_image=True,
            font_path="C:/Windows/Fonts/times.ttf"  # Hoặc arial.ttf
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
        
        print(f"\n💾 Tất cả file output:")
        print(f"- JSON: ./output/translated_ja_to_vi.json")
        print(f"- Ảnh dịch: ./output/{os.path.splitext(os.path.basename(image_path))[0]}_translated.jpg")
        print(f"- Ảnh debug: ./output/{os.path.splitext(os.path.basename(image_path))[0]}_debug_boxes.jpg")
        
        # In chi tiết vài text đầu tiên
        print(f"\n🔍 Chi tiết 3 text đầu tiên:")
        for i, text_data in enumerate(result['texts'][:3]):
            print(f"{i+1}. Gốc: '{text_data['original_text']}'")
            print(f"   Dịch: '{text_data['translated_text']}'")
            print(f"   Confidence: {text_data['confidence']:.3f}")
            print(f"   Merged: {text_data['is_merged']}")
            print()
        
    else:
        print(f"❌ File {image_path} không tồn tại")
        print("Tạo thư mục cropped_boxes/ và đặt ảnh test vào đó")

def test_other_languages():
    """Test với các ngôn ngữ khác"""
    image_path = "./test_images/sample.jpg"  # Thay path phù hợp
    
    if os.path.exists(image_path):
        print("=== TEST CÁC NGÔN NGỮ KHÁC ===")
        
        # Test Trung → Việt
        print("\n🇨🇳➡️🇻🇳 Trung Quốc → Việt Nam")
        process_image_with_translation(
            image_path,
            src_lang="zh",
            dest_lang="vi",
            output_json_path="translated_zh_to_vi.json"
        )
        
        # Test Hàn → Việt  
        print("\n🇰🇷➡️🇻🇳 Hàn Quốc → Việt Nam")
        process_image_with_translation(
            image_path,
            src_lang="ko", 
            dest_lang="vi",
            output_json_path="translated_ko_to_vi.json"
        )
        
        # Test Anh → Việt
        print("\n🇺🇸➡️🇻🇳 Tiếng Anh → Việt Nam") 
        process_image_with_translation(
            image_path,
            src_lang="en",
            dest_lang="vi",
            output_json_path="translated_en_to_vi.json"
        )

def quick_translate_image(image_path, src_lang="ja", dest_lang="vi"):
    """
    Hàm nhanh để dịch ảnh với tham số tối thiểu
    
    Args:
        image_path: Đường dẫn ảnh
        src_lang: Ngôn ngữ nguồn
        dest_lang: Ngôn ngữ đích
        
    Returns:
        str: Đường dẫn ảnh đã dịch
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Ảnh không tồn tại: {image_path}")
    
    print(f"🚀 Dịch nhanh: {src_lang} → {dest_lang}")
    
    result = process_image_with_translation(
        image_path,
        src_lang=src_lang,
        dest_lang=dest_lang,
        save_debug_image=False,
        save_translated_image=True
    )
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    translated_image_path = f"./output/{base_name}_translated.jpg"
    
    print(f"✅ Hoàn thành! Ảnh dịch: {translated_image_path}")
    return translated_image_path

# Các hàm tiện ích
def batch_translate_images(image_folder, src_lang="ja", dest_lang="vi"):
    """
    Dịch hàng loạt ảnh trong folder
    
    Args:
        image_folder: Đường dẫn folder chứa ảnh
        src_lang: Ngôn ngữ nguồn  
        dest_lang: Ngôn ngữ đích
    """
    if not os.path.exists(image_folder):
        print(f"❌ Folder không tồn tại: {image_folder}")
        return
    
    # Tìm tất cả ảnh trong folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(image_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(image_folder, file))
    
    if not image_files:
        print(f"❌ Không tìm thấy ảnh nào trong {image_folder}")
        return
    
    print(f"📁 Tìm thấy {len(image_files)} ảnh để dịch")
    print(f"🌍 Dịch từ {src_lang} sang {dest_lang}")
    
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files):
        try:
            print(f"\n📸 Đang xử lý {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            quick_translate_image(image_path, src_lang, dest_lang)
            successful += 1
            
        except Exception as e:
            print(f"❌ Lỗi xử lý {os.path.basename(image_path)}: {str(e)}")
            failed += 1
    
    print(f"\n📊 Kết quả batch translation:")
    print(f"✅ Thành công: {successful}")
    print(f"❌ Thất bại: {failed}")
    print(f"📁 Tất cả kết quả trong folder: ./output/")

if __name__ == "__main__":
    # Chạy test cơ bản
    test_ocr_translation()
    
    # Uncomment để test các ngôn ngữ khác
    # test_other_languages()
    
    # Uncomment để test batch processing
    # batch_translate_images("./test_images/", "ja", "vi")