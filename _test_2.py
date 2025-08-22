import json
import os
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional, Tuple
import textwrap

def wrap_text_to_fit_bbox(text: str, font, max_width: int, max_height: int) -> List[str]:
    """
    Chia text thành nhiều dòng để vừa với bbox
    
    Args:
        text: Text cần chia
        font: Font object
        max_width: Chiều rộng tối đa
        max_height: Chiều cao tối đa
        
    Returns:
        List[str]: Danh sách các dòng text
    """
    if not text.strip():
        return []
    
    # Thử với số từ khác nhau trên mỗi dòng
    words = text.split()
    if not words:
        return []
    
    # Bắt đầu với 1 từ mỗi dòng và tăng dần
    best_lines = []
    
    for words_per_line in range(1, len(words) + 1):
        lines = []
        current_line = []
        
        for word in words:
            # Thử thêm từ vào dòng hiện tại
            test_line = current_line + [word]
            test_text = ' '.join(test_line)
            
            # Kiểm tra độ rộng của dòng
            bbox = font.getbbox(test_text)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= max_width and len(test_line) <= words_per_line:
                current_line = test_line
            else:
                # Dòng hiện tại đã đầy, chuyển sang dòng mới
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        # Thêm dòng cuối
        if current_line:
            lines.append(' '.join(current_line))
        
        # Kiểm tra chiều cao tổng
        line_height = font.getbbox('Ag')[3] - font.getbbox('Ag')[1]  # Chiều cao 1 dòng
        total_height = len(lines) * line_height * 1.2  # Thêm khoảng cách giữa các dòng
        
        if total_height <= max_height:
            best_lines = lines
        else:
            break  # Không còn vừa chiều cao
    
    return best_lines

def get_optimal_font_size_with_wrapping(text: str, bbox_width: int, bbox_height: int, 
                                      font_path: str, min_size: int = 8, max_size: int = 60) -> Tuple[int, List[str]]:
    """
    Tìm font size tối ưu và chia text thành các dòng vừa với bbox
    
    Args:
        text: Text cần vẽ
        bbox_width: Chiều rộng bbox
        bbox_height: Chiều cao bbox
        font_path: Đường dẫn font
        min_size: Font size tối thiểu
        max_size: Font size tối đa
        
    Returns:
        Tuple[int, List[str]]: (font_size, list_of_lines)
    """
    if not text.strip():
        return min_size, []
    
    # Để lại margin 10%
    margin_width = bbox_width * 0.1
    margin_height = bbox_height * 0.1
    effective_width = bbox_width - margin_width
    effective_height = bbox_height - margin_height
    
    # Binary search để tìm font size tối ưu
    best_size = min_size
    best_lines = [text]  # Fallback
    
    for font_size in range(max_size, min_size - 1, -1):  # Thử từ lớn xuống nhỏ
        try:
            font = ImageFont.truetype(font_path, font_size)
            
            # Thử chia text thành các dòng
            lines = wrap_text_to_fit_bbox(text, font, effective_width, effective_height)
            
            if lines:  # Nếu có thể chia thành công
                # Kiểm tra lại chiều cao tổng
                line_height = font.getbbox('Ag')[3] - font.getbbox('Ag')[1]
                total_height = len(lines) * line_height * 1.2
                
                if total_height <= effective_height:
                    best_size = font_size
                    best_lines = lines
                    break  # Đã tìm được font size tốt nhất
                    
        except Exception:
            continue
    
    return best_size, best_lines

def convert_bbox_format(bbox):
    """
    Chuyển đổi bbox về format [x_min, y_min, x_max, y_max]
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
        else:
            # Fallback
            result = [0, 0, 100, 100]
        
        # Đảm bảo tất cả giá trị là integers và >= 0
        result = [max(0, int(x)) for x in result]
        return result
        
    except Exception:
        return [0, 0, 100, 100]

def draw_multiline_text(draw, lines: List[str], font, x: int, y: int, 
                       text_color: str, line_spacing: float = 1.2):
    """
    Vẽ text nhiều dòng
    
    Args:
        draw: ImageDraw object
        lines: Danh sách các dòng text
        font: Font object
        x, y: Vị trí bắt đầu
        text_color: Màu text
        line_spacing: Khoảng cách giữa các dòng (hệ số)
    """
    if not lines:
        return
    
    line_height = font.getbbox('Ag')[3] - font.getbbox('Ag')[1]
    actual_line_spacing = line_height * line_spacing
    
    for i, line in enumerate(lines):
        line_y = y + i * actual_line_spacing
        draw.text((x, line_y), line, fill=text_color, font=font)

def draw_text_on_image_from_json(image_path, json_path, 
                                font_path="C:/Windows/Fonts/times.ttf",
                                output_path=None,
                                draw_bbox_outline=True,
                                bbox_color_merged="red",  # Đỏ cho merged box
                                bbox_color_single="blue",  # Xanh cho single box
                                bbox_thickness=2,
                                text_color="black",  # Đen
                                bg_color="white"):  # Trắng
    """
    Vẽ text đã dịch lên ảnh gốc dựa vào JSON với hỗ trợ xuống dòng
    
    Args:
        image_path: Đường dẫn ảnh gốc
        json_path: Đường dẫn file JSON chứa kết quả OCR và dịch
        font_path: Đường dẫn font
        output_path: Đường dẫn lưu ảnh kết quả (None = tự động)
        draw_bbox_outline: Có vẽ viền bbox không
        bbox_color_merged: Màu viền cho merged box (tên màu hoặc hex)
        bbox_color_single: Màu viền cho single box (tên màu hoặc hex)
        bbox_thickness: Độ dày viền
        text_color: Màu text (tên màu hoặc hex)
        bg_color: Màu nền của bbox (tên màu hoặc hex)
        
    Returns:
        str: Đường dẫn ảnh đã vẽ text
    """
    # Kiểm tra file tồn tại
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Ảnh không tồn tại: {image_path}")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON không tồn tại: {json_path}")
    
    # Load JSON data
    print(f"📖 Đang đọc JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Load ảnh gốc
    print(f"🖼️ Đang load ảnh: {image_path}")
    original_image = Image.open(image_path)
    img_with_text = original_image.copy()
    draw = ImageDraw.Draw(img_with_text)
    
    # Kiểm tra font
    font_available = os.path.exists(font_path) if font_path else False
    if not font_available:
        print(f"⚠️ Font không tồn tại: {font_path}")
        print("Sử dụng font mặc định")
    
    # Vẽ text cho từng bbox
    text_infos = json_data.get('texts', [])
    print(f"🎨 Đang vẽ {len(text_infos)} text boxes với hỗ trợ xuống dòng...")
    
    success_count = 0
    error_count = 0
    
    for i, text_info in enumerate(text_infos):
        try:
            bbox = text_info.get('bbox', [])
            translated_text = text_info.get('translated_text', text_info.get('original_text', ''))
            is_merged = text_info.get('is_merged', False)
            
            if not translated_text.strip():
                continue
            
            # Convert bbox về format chuẩn
            x_min, y_min, x_max, y_max = convert_bbox_format(bbox)
            
            # Tính kích thước bbox
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            if bbox_width <= 0 or bbox_height <= 0:
                print(f"⚠️ Bbox {i} có kích thước không hợp lệ: {bbox}")
                continue
            
            # Vẽ nền trắng cho bbox
            draw.rectangle([x_min, y_min, x_max, y_max], fill=bg_color, outline=None)
            
            # Tính font size tối ưu và chia text thành dòng
            if font_available:
                optimal_font_size, text_lines = get_optimal_font_size_with_wrapping(
                    translated_text, bbox_width, bbox_height, font_path
                )
                font = ImageFont.truetype(font_path, optimal_font_size)
            else:
                # Sử dụng font mặc định và chia text đơn giản
                try:
                    font = ImageFont.load_default()
                    # Chia text đơn giản theo chiều rộng
                    estimated_chars_per_line = max(1, bbox_width // 8)  # Ước tính
                    text_lines = textwrap.wrap(translated_text, width=estimated_chars_per_line)
                    if not text_lines:
                        text_lines = [translated_text]
                except:
                    print(f"⚠️ Không thể load font cho text {i}")
                    continue
            
            if not text_lines:
                continue
            
            # Tính toán vị trí để center text block
            line_height = font.getbbox('Ag')[3] - font.getbbox('Ag')[1]
            total_text_height = len(text_lines) * line_height * 1.2
            
            # Căn giữa theo chiều dọc
            start_y = y_min + max(0, (bbox_height - total_text_height) // 2)
            
            # Vẽ từng dòng text
            for line_idx, line in enumerate(text_lines):
                if not line.strip():
                    continue
                
                # Tính vị trí x để căn giữa dòng
                line_bbox = font.getbbox(line)
                line_width = line_bbox[2] - line_bbox[0]
                line_x = x_min + max(0, (bbox_width - line_width) // 2)
                
                # Đảm bảo text không vượt ra ngoài bbox
                line_x = max(x_min + 5, min(line_x, x_max - line_width - 5))  # Margin 5px
                line_y = start_y + line_idx * line_height * 1.2
                
                # Kiểm tra không vượt quá bbox
                if line_y + line_height <= y_max - 5:  # Margin bottom 5px
                    draw.text((line_x, line_y), line, fill=text_color, font=font)
            
            # Vẽ viền bbox nếu được yêu cầu
            if draw_bbox_outline:
                outline_color = bbox_color_merged if is_merged else bbox_color_single
                draw.rectangle([x_min, y_min, x_max, y_max], 
                             outline=outline_color, width=bbox_thickness)
            
            success_count += 1
            
        except Exception as e:
            print(f"⚠️ Lỗi vẽ text {i}: {str(e)}")
            error_count += 1
            continue
    
    print(f"✅ Hoàn thành vẽ text với xuống dòng: {success_count} thành công, {error_count} lỗi")
    
    # Tạo đường dẫn output
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_translated.jpg")
    
    # Lưu ảnh
    img_with_text.save(output_path, quality=95)
    print(f"💾 Đã lưu ảnh vào: {output_path}")
    
    return output_path

def batch_draw_from_json_folder(json_folder, images_folder=None, 
                               font_path="C:/Windows/Fonts/times.ttf",
                               output_folder="./output"):
    """
    Vẽ text hàng loạt từ các file JSON trong folder
    
    Args:
        json_folder: Folder chứa các file JSON
        images_folder: Folder chứa ảnh gốc (None = tự động tìm từ JSON)
        font_path: Đường dẫn font
        output_folder: Folder lưu kết quả
        
    Returns:
        dict: Thống kê kết quả
    """
    if not os.path.exists(json_folder):
        raise FileNotFoundError(f"Folder JSON không tồn tại: {json_folder}")
    
    # Tìm tất cả file JSON
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    
    if not json_files:
        print(f"❌ Không tìm thấy file JSON nào trong {json_folder}")
        return {'success': 0, 'failed': 0, 'details': []}
    
    print(f"📁 Tìm thấy {len(json_files)} file JSON")
    
    os.makedirs(output_folder, exist_ok=True)
    
    results = {'success': 0, 'failed': 0, 'details': []}
    
    for i, json_file in enumerate(json_files):
        try:
            print(f"\n📄 Xử lý {i+1}/{len(json_files)}: {json_file}")
            
            json_path = os.path.join(json_folder, json_file)
            
            # Đọc JSON để tìm đường dẫn ảnh
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Tìm ảnh gốc
            if images_folder:
                # Tìm trong folder được chỉ định
                base_name = os.path.splitext(json_file)[0].replace('_translated_ocr_result', '')
                possible_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                image_path = None
                
                for ext in possible_extensions:
                    test_path = os.path.join(images_folder, base_name + ext)
                    if os.path.exists(test_path):
                        image_path = test_path
                        break
                
                if not image_path:
                    print(f"❌ Không tìm thấy ảnh gốc cho {json_file}")
                    results['failed'] += 1
                    results['details'].append({'file': json_file, 'status': 'failed', 'error': 'Image not found'})
                    continue
            else:
                # Lấy từ metadata trong JSON
                image_path = json_data.get('metadata', {}).get('image_path')
                if not image_path or not os.path.exists(image_path):
                    print(f"❌ Đường dẫn ảnh không hợp lệ trong {json_file}")
                    results['failed'] += 1
                    results['details'].append({'file': json_file, 'status': 'failed', 'error': 'Invalid image path'})
                    continue
            
            # Tạo output path
            output_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_translated.jpg"
            output_path = os.path.join(output_folder, output_name)
            
            # Vẽ text
            draw_text_on_image_from_json(
                image_path=image_path,
                json_path=json_path,
                font_path=font_path,
                output_path=output_path,
                draw_bbox_outline=True
            )
            
            results['success'] += 1
            results['details'].append({
                'file': json_file, 
                'status': 'success', 
                'output': output_path,
                'image_source': image_path
            })
            
        except Exception as e:
            print(f"❌ Lỗi xử lý {json_file}: {str(e)}")
            results['failed'] += 1
            results['details'].append({'file': json_file, 'status': 'failed', 'error': str(e)})
    
    print(f"\n📊 Kết quả batch draw:")
    print(f"✅ Thành công: {results['success']}")
    print(f"❌ Thất bại: {results['failed']}")
    print(f"📁 Tất cả output trong: {output_folder}")
    
    return results

def quick_draw_from_json(json_path, image_path=None, font_path="C:/Windows/Fonts/times.ttf"):
    """
    Hàm nhanh để vẽ text từ JSON với tham số tối thiểu
    
    Args:
        json_path: Đường dẫn file JSON
        image_path: Đường dẫn ảnh gốc (None = lấy từ JSON)
        font_path: Đường dẫn font
        
    Returns:
        str: Đường dẫn ảnh đã vẽ text
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON không tồn tại: {json_path}")
    
    # Lấy image_path từ JSON nếu không được cung cấp
    if not image_path:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        image_path = json_data.get('metadata', {}).get('image_path')
        
        if not image_path:
            raise ValueError("Không tìm thấy đường dẫn ảnh trong JSON và không được cung cấp")
    
    print(f"🚀 Vẽ text nhanh từ JSON với hỗ trợ xuống dòng")
    
    output_path = draw_text_on_image_from_json(
        image_path=image_path,
        json_path=json_path,
        font_path=font_path,
        draw_bbox_outline=False  # Không vẽ viền để clean hơn
    )
    
    print(f"✅ Hoàn thành! Ảnh đã vẽ text: {output_path}")
    return output_path

def preview_json_content(json_path, max_texts=5):
    """
    Xem trước nội dung JSON để kiểm tra
    
    Args:
        json_path: Đường dẫn file JSON
        max_texts: Số text tối đa để hiển thị
    """
    if not os.path.exists(json_path):
        print(f"❌ JSON không tồn tại: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    metadata = json_data.get('metadata', {})
    texts = json_data.get('texts', [])
    
    print(f"📖 Xem trước JSON: {json_path}")
    print(f"📊 Metadata:")
    print(f"  - Ảnh gốc: {metadata.get('image_path', 'N/A')}")
    print(f"  - Tổng box: {metadata.get('total_final_boxes', 0)}")
    print(f"  - Ngôn ngữ: {metadata.get('source_language', 'N/A')} → {metadata.get('target_language', 'N/A')}")
    print(f"  - Thời gian xử lý: {metadata.get('processing_timestamp', 'N/A')}")
    
    print(f"\n📝 Text samples (hiển thị {min(max_texts, len(texts))} đầu tiên):")
    for i, text_info in enumerate(texts[:max_texts]):
        print(f"{i+1}. Gốc: '{text_info.get('original_text', '')}'")
        print(f"   Dịch: '{text_info.get('translated_text', '')}'")
        print(f"   Merged: {text_info.get('is_merged', False)}")
        print()

# Test function
def test_draw_from_json():
    """Test function vẽ text từ JSON"""
    json_path = "./output/translated_ja_to_vi.json"
    image_path = "./cropped_boxes/Chap189-Page011.jpg"  # Đường dẫn ảnh gốc
    
    if os.path.exists(json_path):
        print("=== TEST DRAW TEXT FROM JSON WITH WORD WRAPPING ===")
        
        # Xem trước JSON
        preview_json_content(json_path)
        
        # Vẽ text lên ảnh
        print("\n🎨 Vẽ text lên ảnh với hỗ trợ xuống dòng...")
        output_path = draw_text_on_image_from_json(
            image_path=image_path,
            json_path=json_path,
            font_path="C:/Windows/Fonts/times.ttf",
            draw_bbox_outline=True,
            bbox_color_merged="red",  # Đỏ cho merged box
            bbox_color_single="blue",  # Xanh cho single box
        )
        
        print(f"✅ Hoàn thành! Ảnh đã vẽ text: {output_path}")
        
    else:
        print(f"❌ JSON không tồn tại: {json_path}")
        print("Chạy phần OCR trước để tạo JSON")

if __name__ == "__main__":
    # Chạy test
    test_draw_from_json()