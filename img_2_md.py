import cv2
import numpy as np
import os
import json
import time
from typing import List, Dict, Tuple, Any

# Import các module của bạn
from det_rec_preprocess import run_det_rec_preprocess, initialize_ocr
from layout_detection import run_layout_detection, initialize_layout_detector
from test import table_image_to_markdown  # Giả sử function này có trong test.py
from xycut import recursive_xy_cut, points_to_bbox  # Import XY-Cut functions
    
class DocumentProcessor:
    """Class chính để xử lý pipeline tài liệu với layout-aware processing"""
    
    def __init__(self):
        """Khởi tạo các model cần thiết"""
        print("🔧 Khởi tạo các model...")
        load_start = time.time()
        
        self.ocr_model = initialize_ocr()
        self.layout_model = initialize_layout_detector()
        
        load_time = time.time() - load_start
        print(f"✅ Hoàn thành khởi tạo! Thời gian load model: {load_time:.2f}s")
        print("🚀 Sẵn sàng xử lý nhiều tài liệu!\n")
    
    def is_point_in_box(self, point, box):
        """Kiểm tra điểm có nằm trong box không"""
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def get_text_box_center(self, box):
        """Lấy tọa độ trung tâm của text bounding box"""
        if isinstance(box, list) and len(box) == 4:
            if isinstance(box[0], list):
                x_coords = [point[0] for point in box]
                y_coords = [point[1] for point in box]
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                return center_x, center_y
            else:
                x1, y1, x2, y2 = box
                return (x1 + x2) / 2, (y1 + y2) / 2
        elif hasattr(box, 'shape') and len(box.shape) == 2:
            center_x = np.mean(box[:, 0])
            center_y = np.mean(box[:, 1])
            return center_x, center_y
        else:
            try:
                box_array = np.array(box)
                if box_array.shape == (4, 2):
                    center_x = np.mean(box_array[:, 0])
                    center_y = np.mean(box_array[:, 1])
                    return center_x, center_y
            except:
                pass
            return 0, 0
    
    def classify_texts_by_layout_regions(self, all_texts, layout_regions):
        """
        Phân loại text theo các layout regions (table, image, text)
        
        Returns:
            dict: {
                'free_texts': [],  # Text không thuộc vùng nào đặc biệt
                'texts_in_tables': [],  # Text trong table regions
                'texts_in_images': []   # Text trong image regions (sẽ bị bỏ qua)
            }
        """
        free_texts = []
        texts_in_tables = []
        texts_in_images = []
        
        for text_info in all_texts:
            bbox = text_info.get('bbox', None)
            if not bbox:
                free_texts.append(text_info)
                continue
            
            # Lấy tọa độ trung tâm của text box
            center_x, center_y = self.get_text_box_center(bbox)
            
            # Kiểm tra xem center có nằm trong layout region nào không
            assigned = False
            
            for layout_region in layout_regions:
                region_bbox = layout_region['bbox']  # [x_min, y_min, x_max, y_max]
                
                if self.is_point_in_box((center_x, center_y), region_bbox):
                    text_info['layout_region'] = layout_region
                    
                    # Phân loại theo label của region
                    if layout_region['label'] == 'table':
                        texts_in_tables.append(text_info)
                    elif layout_region['label'] == 'image':
                        texts_in_images.append(text_info)
                    else:
                        # Các label khác (title, text, etc.) vẫn coi là free text
                        free_texts.append(text_info)
                    
                    assigned = True
                    break
            
            if not assigned:
                free_texts.append(text_info)
        
        return {
            'free_texts': free_texts,
            'texts_in_tables': texts_in_tables,
            'texts_in_images': texts_in_images
        }
    
    def save_image_region(self, image_region, output_dir, base_name, region_index):
        """Lưu image region và trả về path"""
        image_filename = f"{base_name}_image_{region_index + 1}.jpg"
        image_path = os.path.join(output_dir, image_filename)
        
        # Lưu ảnh
        cv2.imwrite(image_path, image_region['image'])
        
        return image_filename, image_path
    
    def create_text_blocks_from_free_texts(self, free_texts, layout_regions):
        """
        Tạo các text blocks từ free texts, sắp xếp theo XY-Cut và tránh overlap với layout regions
        """
        if not free_texts:
            return []
        
        # Sử dụng XY-Cut để sắp xếp free texts theo thứ tự đọc
        sorted_free_texts = self.sort_texts_with_xycut(free_texts)
        
        # Group texts thành các blocks dựa trên khoảng cách và vị trí
        text_blocks = []
        current_block = []
        current_y = None
        y_threshold = 50  # Threshold để group texts trong cùng một "paragraph"
        
        for text_info in sorted_free_texts:
            bbox = text_info.get('bbox', [[0, 0], [0, 0], [0, 0], [0, 0]])
            _, center_y = self.get_text_box_center(bbox)
            
            if current_y is None or abs(center_y - current_y) > y_threshold:
                # Bắt đầu block mới
                if current_block:
                    text_blocks.append(current_block)
                current_block = [text_info]
                current_y = center_y
            else:
                # Thêm vào block hiện tại
                current_block.append(text_info)
        
        # Đừng quên block cuối
        if current_block:
            text_blocks.append(current_block)
        
        return text_blocks

    def sort_texts_by_position(self, texts):
        """Sắp xếp text theo vị trí từ trên xuống dưới, trái sang phải (fallback method)"""
        def get_sort_key(text_info):
            bbox = text_info.get('bbox', [[0, 0], [0, 0], [0, 0], [0, 0]])
            center_x, center_y = self.get_text_box_center(bbox)
            return (center_y, center_x)
        
        return sorted(texts, key=get_sort_key)
    
    def process_document(self, image_path: str, output_dir: str = "./output/") -> str:
        """
        Xử lý toàn bộ pipeline từ ảnh đầu vào đến markdown cuối với layout-aware processing
        """
        # Tạo thư mục output
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        print(f"🚀 Bắt đầu xử lý: {image_path}")
        process_start_time = time.time()
        
        # Bước 1: Tiền xử lý ảnh với OCR
        print("📝 Bước 1: Chạy det_rec_preprocess...")
        step1_start = time.time()
        
        processed_image, all_texts, all_boxes = run_det_rec_preprocess(
            image_path, 
            self.ocr_model
        )
        
        step1_time = time.time() - step1_start
        print(f"   ⏱️ Thời gian bước 1: {step1_time:.2f}s")
        print(f"   📝 Tổng số text được detect: {len(all_texts)}")
        
        # Lưu ảnh đã tiền xử lý
        processed_image_path = os.path.join(output_dir, f"{base_name}_processed.jpg")
        cv2.imwrite(processed_image_path, processed_image)
        
        # Bước 2: Layout detection
        print("🔍 Bước 2: Chạy layout_detection...")
        step2_start = time.time()
        
        original_img, layout_regions, layout_boxes = run_layout_detection(
            processed_image_path, 
            self.layout_model
        )
        
        # Phân loại layout regions
        table_regions = [region for region in layout_regions if region['label'] == 'table']
        image_regions = [region for region in layout_regions if region['label'] == 'image']
        other_regions = [region for region in layout_regions if region['label'] not in ['table', 'image']]
        
        step2_time = time.time() - step2_start
        print(f"   ⏱️ Thời gian bước 2: {step2_time:.2f}s")
        print(f"   📊 Layout detection kết quả:")
        print(f"      - Table regions: {len(table_regions)}")
        print(f"      - Image regions: {len(image_regions)}")
        print(f"      - Other regions: {len(other_regions)}")
        
        # Bước 3: Phân loại text theo layout regions
        print("⚙️ Bước 3: Phân loại text theo layout regions...")
        step3_start = time.time()
        
        text_classification = self.classify_texts_by_layout_regions(all_texts, layout_regions)
        
        print(f"   📝 Text classification:")
        print(f"      - Free texts: {len(text_classification['free_texts'])}")
        print(f"      - Texts in tables: {len(text_classification['texts_in_tables'])}")
        print(f"      - Texts in images (ignored): {len(text_classification['texts_in_images'])}")
        
        # Bước 4: Tạo các content sections với đúng vị trí
        content_sections = []
        
        # 4.1: Xử lý table regions
        print(f"   🔄 Xử lý {len(table_regions)} table regions...")
        for i, table_region in enumerate(table_regions):
            try:
                table_markdown = table_image_to_markdown(table_region['image'])
                content_sections.append({
                    'type': 'table',
                    'content': table_markdown,
                    'bbox': table_region['bbox'],
                    'y_position': table_region['bbox'][1],  # y_min để sắp xếp
                    'index': i + 1
                })
                print(f"      ✅ Table {i+1} processed")
            except Exception as e:
                print(f"      ❌ Lỗi khi xử lý table {i+1}: {str(e)}")
        
        # 4.2: Xử lý image regions
        print(f"   🖼️ Xử lý {len(image_regions)} image regions...")
        for i, image_region in enumerate(image_regions):
            try:
                image_filename, image_path = self.save_image_region(
                    image_region, output_dir, base_name, i
                )
                content_sections.append({
                    'type': 'image',
                    'content': image_filename,
                    'bbox': image_region['bbox'],
                    'y_position': image_region['bbox'][1],  # y_min để sắp xếp
                    'index': i + 1
                })
                print(f"      ✅ Image {i+1} saved as {image_filename}")
            except Exception as e:
                print(f"      ❌ Lỗi khi xử lý image {i+1}: {str(e)}")
        
        # 4.3: Xử lý free text blocks với XY-Cut
        free_texts = text_classification['free_texts']
        if free_texts:
            print(f"   📄 Xử lý {len(free_texts)} free texts với XY-Cut...")
            text_blocks = self.create_text_blocks_from_free_texts(free_texts, layout_regions)
            print(f"   📦 Tạo được {len(text_blocks)} text blocks")
            
            for i, text_block in enumerate(text_blocks):
                # Tính vị trí trung bình của block
                y_positions = []
                for text_info in text_block:
                    bbox = text_info.get('bbox', [[0, 0], [0, 0], [0, 0], [0, 0]])
                    _, center_y = self.get_text_box_center(bbox)
                    y_positions.append(center_y)
                
                avg_y = sum(y_positions) / len(y_positions) if y_positions else 0
                
                # Combine texts trong block theo thứ tự XY-Cut
                combined_text = self.combine_texts_to_paragraphs(text_block)
                
                if combined_text.strip():
                    content_sections.append({
                        'type': 'text',
                        'content': combined_text,
                        'bbox': [0, avg_y, 1000, avg_y + 50],  # Dummy bbox
                        'y_position': avg_y,
                        'index': i + 1
                    })
        
        step3_time = time.time() - step3_start
        print(f"   ⏱️ Thời gian bước 3-4: {step3_time:.2f}s")
        
        # Bước 5: Sắp xếp và tạo markdown cuối
        print("📋 Bước 5: Tạo markdown với layout đúng vị trí...")
        step5_start = time.time()
        
        markdown_content = self.create_layout_aware_markdown(
            content_sections, image_path, base_name
        )
        
        # Lưu file markdown
        markdown_path = os.path.join(output_dir, f"{base_name}_result.md")
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        step5_time = time.time() - step5_start
        print(f"   ⏱️ Thời gian bước 5: {step5_time:.2f}s")
        
        # Tổng thời gian xử lý
        total_process_time = time.time() - process_start_time
        print(f"⏱️ Tổng thời gian xử lý: {total_process_time:.2f}s")
        print(f"✅ Hoàn thành! Markdown saved: {markdown_path}\n")
        
        return markdown_path
    
    def process_multiple_documents(self, image_paths: List[str], output_dir: str = "./output/") -> List[str]:
        """Xử lý nhiều tài liệu cùng lúc với cùng một model đã load"""
        results = []
        total_start_time = time.time()
        
        print(f"📚 Bắt đầu xử lý {len(image_paths)} tài liệu...")
        print(f"{'='*60}")
        
        for i, image_path in enumerate(image_paths, 1):
            if not os.path.exists(image_path):
                print(f"⚠️ File không tồn tại: {image_path}")
                continue
                
            print(f"\n📄 Tài liệu {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            print("-" * 50)
            
            try:
                result_path = self.process_document(image_path, output_dir)
                results.append(result_path)
                
                # Hiển thị preview nội dung
                with open(result_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"📖 Preview markdown:")
                    preview = content[:300] + "..." if len(content) > 300 else content
                    print(preview)
                    print("-" * 30)
                    
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {image_path}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        total_time = time.time() - total_start_time
        print(f"\n🏁 Hoàn thành tất cả!")
        print(f"📊 Thống kê:")
        print(f"   - Tổng thời gian: {total_time:.2f}s")
        print(f"   - Số tài liệu thành công: {len(results)}/{len(image_paths)}")
        print(f"   - Thời gian trung bình/tài liệu: {total_time/max(len(results), 1):.2f}s")
        print(f"   - Kết quả lưu tại: {output_dir}")
        
        return results
    
    def convert_bbox_format(self, bbox):
        """
        Chuyển đổi bbox từ các format khác nhau về format [x_min, y_min, x_max, y_max]
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
                # Fallback
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
            
        except Exception as e:
            print(f"      ⚠️ Error converting bbox format: {e}")
            return [0, 0, 100, 100]  # Safe fallback

    def sort_texts_with_xycut(self, texts):
        """Sắp xếp text theo thứ tự đọc sử dụng XY-Cut algorithm"""
        if not texts:
            return []
        
        # Chuẩn bị dữ liệu cho XY-Cut
        boxes = []
        indices = list(range(len(texts)))
        
        for i, text_info in enumerate(texts):
            bbox = text_info.get('bbox', [[0, 0], [0, 0], [0, 0], [0, 0]])
            # Chuyển về format [x_min, y_min, x_max, y_max]
            converted_bbox = self.convert_bbox_format(bbox)
            boxes.append(converted_bbox)
        
        boxes = np.array(boxes, dtype=np.int32)  # Đảm bảo dtype là int32
        indices = np.array(indices, dtype=np.int32)  # Chuyển indices thành numpy array
        
        # Kiểm tra dữ liệu đầu vào
        if len(boxes) == 0:
            return []
        
        try:
            # Chạy XY-Cut để lấy thứ tự đọc
            reading_order = []
            recursive_xy_cut(boxes, indices, reading_order)
            
            # Sắp xếp texts theo thứ tự đọc
            if reading_order:
                sorted_texts = [texts[i] for i in reading_order]
            else:
                # Fallback về sắp xếp đơn giản nếu XY-Cut không trả về kết quả
                print("      ⚠️ XY-Cut không trả về kết quả, sử dụng fallback sorting")
                sorted_texts = self.sort_texts_by_position(texts)
                
        except Exception as e:
            print(f"      ⚠️ XY-Cut error: {str(e)}, sử dụng fallback sorting")
            # Fallback về sắp xếp đơn giản nếu XY-Cut thất bại
            sorted_texts = self.sort_texts_by_position(texts)
        
        return sorted_texts
    
    def combine_texts_to_paragraphs(self, sorted_texts):
        """Kết hợp texts thành paragraphs với thứ tự đọc đúng từ XY-Cut"""
        if not sorted_texts:
            return ""

        # Với XY-Cut, texts đã được sắp xếp theo thứ tự đọc tự nhiên
        # Chỉ cần join với space để tạo thành đoạn văn liền mạch
        lines = []
        for text_info in sorted_texts:
            text = text_info.get('text', '').strip()
            if text:
                lines.append(text)

        return ' '.join(lines)  # Join với space cho đoạn văn tự nhiên
    
    def create_layout_aware_markdown(self, content_sections, original_image_path, base_name):
        """Tạo markdown đơn giản, hiển thị giống ảnh đầu vào"""
        markdown_lines = []
        
        # Sắp xếp content sections theo y_position (từ trên xuống dưới)
        sorted_sections = sorted(content_sections, key=lambda x: x['y_position'])
        
        # Render từng section theo đúng thứ tự, không thêm header thừa
        for i, section in enumerate(sorted_sections):
            section_type = section['type']
            content = section['content']
            
            if section_type == 'text':
                # Chỉ thêm content, không có header
                if content.strip():
                    markdown_lines.append(content.strip())
                    
            elif section_type == 'table':
                # Chỉ thêm table, không có header
                if content.strip():
                    markdown_lines.append(content.strip())
                    
            elif section_type == 'image':
                # Chỉ thêm image, không có header
                markdown_lines.append(f"![Image](./{content})")
            
            # Thêm khoảng trắng giữa các sections (trừ section cuối)
            if i < len(sorted_sections) - 1:
                markdown_lines.append("")
        
        return '\n\n'.join(markdown_lines)

# UTILITY FUNCTIONS

def process_single_document(image_path: str, output_dir: str = "./output/") -> str:
    """Hàm tiện ích để xử lý một tài liệu đơn lẻ"""
    processor = DocumentProcessor()
    return processor.process_document(image_path, output_dir)

def process_multiple_documents_optimized(image_paths: List[str], output_dir: str = "./output/") -> List[str]:
    """Hàm tiện ích để xử lý nhiều tài liệu với model chỉ load một lần"""
    processor = DocumentProcessor()  # Load model chỉ một lần
    return processor.process_multiple_documents(image_paths, output_dir)

# CÁCH SỬ DỤNG:

if __name__ == "__main__":
    print("=== XỬ LÝ TÀI LIỆU VỚI LAYOUT-AWARE PROCESSING ===")
    
    # Danh sách các ảnh test
    test_images = [
        "./anh_test/1.jpg",
        "./anh_test/2.jpg", 
        "./anh_test/3.jpg",
        "./anh_test/4.jpg",
        "./anh_test/5.jpg",
        "./anh_test/6.jpg",
        "./anh_test/7.jpg",
        "./anh_test/8.jpg",
        "./anh_test/9.jpg",
        "./anh_test/10.jpg",
        "./anh_test/11.jpg",
    ]
    
    # Lọc chỉ các file tồn tại
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if not existing_images:
        print("❌ Không tìm thấy file ảnh nào!")
        exit()
    
    print(f"✅ Tìm thấy {len(existing_images)} file ảnh hợp lệ")
    
    # Tạo processor một lần, xử lý nhiều ảnh
    processor = DocumentProcessor()  # Load model chỉ một lần ở đây!
    results = processor.process_multiple_documents(existing_images, "./output/")
    
    print(f"\n🎉 Hoàn thành! Đã tạo {len(results)} file markdown:")
    for result in results:
        print(f"   📄 {result}")