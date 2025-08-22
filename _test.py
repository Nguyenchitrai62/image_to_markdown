import cv2
import numpy as np
import os
import json
import time
from typing import List, Dict, Tuple, Any, Optional

# Import các module của bạn
from det_rec_preprocess import run_det_rec_preprocess, initialize_ocr
from layout_detection import run_layout_detection, initialize_layout_detector
from table_procesing import table_image_to_markdown, initialize_cell_detector
from ori import run_orientation_correction, initialize_orientation_classifier
from xycut import recursive_xy_cut, points_to_bbox

class ProcessingConfig:
    """Configuration class để điều khiển việc bật/tắt các module"""
    
    def __init__(self,
                 orientation: bool = True,
                 layout_detection: bool = True,
                 table_processing: bool = True,
                 text_processing: bool = True,
                 image_extraction: bool = True,
                 xy_cut_sorting: bool = True):
        """
        Args:
            orientation: Bật/tắt xác định hướng ảnh
            layout_detection: Bật/tắt phát hiện layout
            table_processing: Bật/tắt xử lý bảng
            text_processing: Bật/tắt xử lý văn bản
            image_extraction: Bật/tắt trích xuất ảnh
            xy_cut_sorting: Bật/tắt sắp xếp XY-Cut (fallback về sắp xếp đơn giản)
        """
        self.orientation = orientation
        self.layout_detection = layout_detection
        self.table_processing = table_processing
        self.text_processing = text_processing
        self.image_extraction = image_extraction
        self.xy_cut_sorting = xy_cut_sorting
        
    @classmethod
    def minimal_config(cls):
        """Config tối thiểu - chỉ OCR cơ bản"""
        return cls(
            orientation=False,
            layout_detection=False,
            table_processing=False,
            text_processing=True,
            image_extraction=False,
            xy_cut_sorting=False
        )
    
    @classmethod
    def full_config(cls):
        """Config đầy đủ - tất cả modules"""
        return cls(
            orientation=True,
            layout_detection=True,
            table_processing=True,
            text_processing=True,
            image_extraction=True,
            xy_cut_sorting=True
        )
    
    @classmethod
    def text_only_config(cls):
        """Config chỉ xử lý text - không xử lý bảng/ảnh"""
        return cls(
            orientation=True,
            layout_detection=False,
            table_processing=False,
            text_processing=True,
            image_extraction=False,
            xy_cut_sorting=True
        )

class DocumentProcessor:
    """Class chính để xử lý pipeline tài liệu với layout-aware processing"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Khởi tạo các model cần thiết dựa trên config
        
        Args:
            config: ProcessingConfig để điều khiển việc load model nào
        """
        if config is None:
            config = ProcessingConfig.full_config()
        
        self.config = config
        
        load_start = time.time()
        print("Đang khởi tạo các model...")
        
        # Luôn khởi tạo OCR model (bắt buộc)
        print("- Đang khởi tạo OCR model...")
        self.ocr_model = initialize_ocr()
        
        # Khởi tạo Orientation Classifier model (tùy chọn)
        self.orientation_model = None
        if self.config.orientation:
            print("- Đang khởi tạo Orientation Classifier model...")
            self.orientation_model = initialize_orientation_classifier()
        else:
            print("- Bỏ qua Orientation Classifier model")
        
        # Khởi tạo Layout Detection model (tùy chọn)
        self.layout_model = None
        if self.config.layout_detection:
            print("- Đang khởi tạo Layout Detection model...")
            self.layout_model = initialize_layout_detector()
        else:
            print("- Bỏ qua Layout Detection model")
        
        # Khởi tạo Cell Detection model (tùy chọn)
        self.cell_model = None
        if self.config.table_processing:
            print("- Đang khởi tạo Cell Detection model...")
            self.cell_model = initialize_cell_detector()
        else:
            print("- Bỏ qua Cell Detection model")
        
        load_time = time.time() - load_start
        print(f"Các model đã được khởi tạo thành công trong {load_time:.2f}s")
    
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
        # Nếu không có layout detection, tất cả text đều là free text
        if not self.config.layout_detection or not layout_regions:
            return {
                'free_texts': all_texts,
                'texts_in_tables': [],
                'texts_in_images': []
            }
        
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
        
        # Sử dụng XY-Cut hoặc sắp xếp đơn giản tùy theo config
        if self.config.xy_cut_sorting:
            sorted_free_texts = self.sort_texts_with_xycut(free_texts)
        else:
            sorted_free_texts = self.sort_texts_by_position(free_texts)
        
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
        
        process_start_time = time.time()
        timing_info = {}
        
        # Bước 0: Orientation correction (TÙY CHỌN)
        if self.config.orientation and self.orientation_model is not None:
            step0_start = time.time()
            
            corrected_image, orientation_info = run_orientation_correction(
                image_path, 
                self.orientation_model
            )
            
            timing_info['orientation'] = time.time() - step0_start
            print(f"✓ Orientation correction: {timing_info['orientation']:.2f}s")
        else:
            # Đọc ảnh trực tiếp nếu không xử lý orientation
            corrected_image = cv2.imread(image_path)
            timing_info['orientation'] = 0.0
            print("⊝ Bỏ qua orientation correction")
        
        # Bước 1: Tiền xử lý ảnh với OCR (LUÔN LUÔN CHẠY)
        step1_start = time.time()
        
        processed_image, all_texts, all_boxes = run_det_rec_preprocess(
            corrected_image,  # Sử dụng numpy array ảnh
            self.ocr_model
        )
        
        timing_info['ocr'] = time.time() - step1_start
        print(f"✓ OCR processing: {timing_info['ocr']:.2f}s")
        
        # Lưu ảnh đã tiền xử lý
        processed_image_path = os.path.join(output_dir, f"{base_name}_processed.jpg")
        cv2.imwrite(processed_image_path, processed_image)
        
        # Bước 2: Layout detection (TÙY CHỌN)
        layout_regions = []
        if self.config.layout_detection and self.layout_model is not None:
            step2_start = time.time()
            
            original_img, layout_regions, layout_boxes = run_layout_detection(
                processed_image_path, 
                self.layout_model
            )
            
            timing_info['layout'] = time.time() - step2_start
            print(f"✓ Layout detection: {timing_info['layout']:.2f}s")
        else:
            timing_info['layout'] = 0.0
            print("⊝ Bỏ qua layout detection")
        
        # Phân loại layout regions
        table_regions = [region for region in layout_regions if region['label'] == 'table'] if self.config.table_processing else []
        image_regions = [region for region in layout_regions if region['label'] == 'image'] if self.config.image_extraction else []
        other_regions = [region for region in layout_regions if region['label'] not in ['table', 'image']]
        
        # Bước 3: Phân loại text theo layout regions và tạo content
        step3_start = time.time()
        
        text_classification = self.classify_texts_by_layout_regions(all_texts, layout_regions)
        
        # Bước 4: Tạo các content sections với đúng vị trí
        content_sections = []
        
        # 4.1: Xử lý table regions (TÙY CHỌN)
        if self.config.table_processing and self.cell_model is not None:
            for i, table_region in enumerate(table_regions):
                try:
                    table_markdown = table_image_to_markdown(
                        table_region['image'], 
                        cell_model=self.cell_model
                    )
                    content_sections.append({
                        'type': 'table',
                        'content': table_markdown,
                        'bbox': table_region['bbox'],
                        'y_position': table_region['bbox'][1],
                        'index': i + 1
                    })
                except Exception as e:
                    print(f"Lỗi khi xử lý table {i+1}: {e}")
                    content_sections.append({
                        'type': 'table',
                        'content': '<table border="1"><tr><td>Error processing table</td></tr></table>',
                        'bbox': table_region['bbox'],
                        'y_position': table_region['bbox'][1],
                        'index': i + 1
                    })
        
        # 4.2: Xử lý image regions (TÙY CHỌN)
        if self.config.image_extraction:
            for i, image_region in enumerate(image_regions):
                try:
                    image_filename, image_path = self.save_image_region(
                        image_region, output_dir, base_name, i
                    )
                    content_sections.append({
                        'type': 'image',
                        'content': image_filename,
                        'bbox': image_region['bbox'],
                        'y_position': image_region['bbox'][1],
                        'index': i + 1
                    })
                except Exception as e:
                    print(f"Lỗi khi xử lý image {i+1}: {e}")
        
        # 4.3: Xử lý free text blocks (TÙY CHỌN)
        if self.config.text_processing:
            free_texts = text_classification['free_texts']
            if free_texts:
                text_blocks = self.create_text_blocks_from_free_texts(free_texts, layout_regions)
                
                for i, text_block in enumerate(text_blocks):
                    # Tính vị trí trung bình của block
                    y_positions = []
                    for text_info in text_block:
                        bbox = text_info.get('bbox', [[0, 0], [0, 0], [0, 0], [0, 0]])
                        _, center_y = self.get_text_box_center(bbox)
                        y_positions.append(center_y)
                    
                    avg_y = sum(y_positions) / len(y_positions) if y_positions else 0
                    
                    # Combine texts trong block
                    combined_text = self.combine_texts_to_paragraphs(text_block)
                    
                    if combined_text.strip():
                        content_sections.append({
                            'type': 'text',
                            'content': combined_text,
                            'bbox': [0, avg_y, 1000, avg_y + 50],
                            'y_position': avg_y,
                            'index': i + 1
                        })
        
        timing_info['processing'] = time.time() - step3_start
        
        # Bước 5: Sắp xếp và tạo markdown cuối
        step5_start = time.time()
        
        markdown_content = self.create_layout_aware_markdown(
            content_sections, image_path, base_name
        )
        
        # Lưu file markdown
        markdown_path = os.path.join(output_dir, f"{base_name}_result.md")
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        timing_info['markdown'] = time.time() - step5_start
        
        # Tổng thời gian xử lý
        total_process_time = time.time() - process_start_time
        
        # In thống kê chi tiết
        print(f"{base_name} - Orientation: {timing_info.get('orientation', 0):.2f}s | "
              f"OCR: {timing_info['ocr']:.2f}s | "
              f"Layout: {timing_info.get('layout', 0):.2f}s | "
              f"Processing: {timing_info['processing']:.2f}s | "
              f"Markdown: {timing_info['markdown']:.2f}s | "
              f"Total: {total_process_time:.2f}s")
        
        return markdown_path
    
    def process_multiple_documents(self, image_paths: List[str], output_dir: str = "./output/") -> List[str]:
        """Xử lý nhiều tài liệu cùng lúc với cùng một model đã load"""
        results = []
        total_start_time = time.time()
        
        for i, image_path in enumerate(image_paths, 1):
            if not os.path.exists(image_path):
                print(f"File không tồn tại: {image_path}")
                continue
                
            try:
                print(f"Đang xử lý ({i}/{len(image_paths)}): {os.path.basename(image_path)}")
                result_path = self.process_document(image_path, output_dir)
                results.append(result_path)
                    
            except Exception as e:
                print(f"Lỗi khi xử lý {image_path}: {e}")
                import traceback
                traceback.print_exc()
        
        total_time = time.time() - total_start_time
        print(f"Total processing time: {total_time:.2f}s | Documents: {len(results)}/{len(image_paths)} | Average: {total_time/max(len(results), 1):.2f}s per document")
        
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
                sorted_texts = self.sort_texts_by_position(texts)
                
        except Exception as e:
            print(f"XY-Cut failed, using fallback sorting: {e}")
            # Fallback về sắp xếp đơn giản nếu XY-Cut thất bại
            sorted_texts = self.sort_texts_by_position(texts)
        
        return sorted_texts
    
    def combine_texts_to_paragraphs(self, sorted_texts):
        """Kết hợp texts thành paragraphs với thứ tự đọc đúng"""
        if not sorted_texts:
            return ""

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
                next_type = sorted_sections[i+1]['type']
                # Chỉ thêm dòng trống nếu loại khác nhau
                if section_type != next_type:
                    markdown_lines.append("")
        
        return '\n'.join(markdown_lines)

# UTILITY FUNCTIONS

def process_single_document(image_path: str, 
                          output_dir: str = "./output/",
                          config: Optional[ProcessingConfig] = None) -> str:
    """
    Hàm tiện ích để xử lý một tài liệu đơn lẻ
    
    Args:
        image_path: Đường dẫn ảnh
        output_dir: Thư mục output
        config: Cấu hình xử lý (None = full config)
    """
    processor = DocumentProcessor(config)
    return processor.process_document(image_path, output_dir)

def process_multiple_documents_optimized(image_paths: List[str], 
                                       output_dir: str = "./output/",
                                       config: Optional[ProcessingConfig] = None) -> List[str]:
    """
    Hàm tiện ích để xử lý nhiều tài liệu với model chỉ load một lần
    
    Args:
        image_paths: List đường dẫn ảnh
        output_dir: Thư mục output  
        config: Cấu hình xử lý (None = full config)
    """
    processor = DocumentProcessor(config)  # Load model chỉ một lần
    return processor.process_multiple_documents(image_paths, output_dir)

# CÁCH SỬ DỤNG:

if __name__ == "__main__":
    # Danh sách các ảnh test
    test_images = [
        # "./anh_test/1.jpg",
        # "./anh_test/2.jpg", 
        # "./anh_test/3.jpg",
        # "./anh_test/4.jpg",
        # "./anh_test/5.jpg",
        # "./anh_test/6.jpg",
        # "./anh_test/7.jpg",
        # "./anh_test/8.jpg",
        # "./anh_test/9.jpg",
        # "./anh_test/10.jpg",
        # "./anh_test/11.jpg",
        "./cropped_boxes/1-6558c6a907fd2.jpg"
    ]
    
    # Lọc chỉ các file tồn tại
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if not existing_images:
        print("No valid image files found!")
        exit()
    
    print(f"Tìm thấy {len(existing_images)} ảnh để xử lý")
    
    # ===== CÁC CÁCH SỬ DỤNG =====
    
    # # Cách 1: Sử dụng config đầy đủ (mặc định)
    # print("\n=== FULL CONFIG ===")
    # config_full = ProcessingConfig.full_config()
    # processor_full = DocumentProcessor(config_full)
    # results_full = processor_full.process_multiple_documents(existing_images[:2], "./output/full/")
    
    # # Cách 2: Chỉ OCR cơ bản (nhanh nhất)
    # print("\n=== MINIMAL CONFIG (OCR only) ===")
    # config_minimal = ProcessingConfig.minimal_config()
    # processor_minimal = DocumentProcessor(config_minimal)
    # results_minimal = processor_minimal.process_multiple_documents(existing_images[:2], "./output/minimal/")
    
    # # Cách 3: Chỉ xử lý text (không table/image)
    # print("\n=== TEXT ONLY CONFIG ===")
    # config_text_only = ProcessingConfig.text_only_config()
    # processor_text = DocumentProcessor(config_text_only)
    # results_text = processor_text.process_multiple_documents(existing_images[:2], "./output/text_only/")
    
    # Cách 4: Custom config - tùy chỉnh từng module
    print("\n=== CUSTOM CONFIG ===")
    config_custom = ProcessingConfig(
        orientation=False,           # Bật xác định hướng
        layout_detection=False,      # Bật phát hiện layout
        table_processing=False,     # TẮT xử lý bảng (nếu không cần)
        text_processing=True,       # Bật xử lý text
        image_extraction=True,      # Bật trích xuất ảnh
        xy_cut_sorting=True         # Bật sắp xếp XY-Cut
    )
    processor_custom = DocumentProcessor(config_custom)
    results_custom = processor_custom.process_multiple_documents(existing_images[:2], "./output/custom/")
    
    # # Cách 5: Xử lý từng file với config khác nhau
    # print("\n=== INDIVIDUAL PROCESSING ===")
    
    # # File 1: Full processing
    # if len(existing_images) > 0:
    #     result1 = process_single_document(
    #         existing_images[0], 
    #         "./output/individual/", 
    #         ProcessingConfig.full_config()
    #     )
    #     print(f"File 1 (full): {result1}")
    
    # # File 2: Text only
    # if len(existing_images) > 1:
    #     result2 = process_single_document(
    #         existing_images[1], 
    #         "./output/individual/", 
    #         ProcessingConfig.text_only_config()
    #     )
    #     print(f"File 2 (text only): {result2}")
    
    # # File 3: Minimal
    # if len(existing_images) > 2:
    #     result3 = process_single_document(
    #         existing_images[2], 
    #         "./output/individual/", 
    #         ProcessingConfig.minimal_config()
    #     )
    #     print(f"File 3 (minimal): {result3}")
    
    print(f"\nHoàn thành! Kiểm tra các thư mục output để xem kết quả.")
    print("- ./output/full/ : Xử lý đầy đủ tất cả modules")
    print("- ./output/minimal/ : Chỉ OCR cơ bản")
    print("- ./output/text_only/ : Chỉ xử lý text (không table/image)")
    print("- ./output/custom/ : Cấu hình tùy chỉnh")
    print("- ./output/individual/ : Xử lý từng file với config khác nhau")

#     # ===== BENCHMARK PERFORMANCE =====
#     print("\n=== PERFORMANCE COMPARISON ===")
    
#     if len(existing_images) > 0:
#         test_image = existing_images[0]
        
#         # Test minimal config
#         import time
        
#         start = time.time()
#         process_single_document(test_image, "./output/benchmark/", ProcessingConfig.minimal_config())
#         minimal_time = time.time() - start
        
#         start = time.time()
#         process_single_document(test_image, "./output/benchmark/", ProcessingConfig.full_config())
#         full_time = time.time() - start
        
#         print(f"Minimal config: {minimal_time:.2f}s")
#         print(f"Full config: {full_time:.2f}s")
#         print(f"Speed improvement: {full_time/minimal_time:.1f}x faster với minimal config")

# # ===== CÁC CONFIG MẪU THÊM =====

# class ProcessingConfigExtended(ProcessingConfig):
#     """Extended config với nhiều tùy chọn hơn"""
    
#     @classmethod
#     def document_scan_config(cls):
#         """Config cho scan tài liệu văn phòng - ưu tiên text và table"""
#         return cls(
#             orientation=True,
#             layout_detection=True,
#             table_processing=True,
#             text_processing=True,
#             image_extraction=False,  # Ít quan trọng với doc scan
#             xy_cut_sorting=True
#         )
    
#     @classmethod
#     def magazine_config(cls):
#         """Config cho tạp chí/brochure - ưu tiên layout và image"""
#         return cls(
#             orientation=True,
#             layout_detection=True,
#             table_processing=False,  # Ít table trong tạp chí
#             text_processing=True,
#             image_extraction=True,   # Quan trọng với tạp chí
#             xy_cut_sorting=True
#         )
    
#     @classmethod
#     def speed_optimized_config(cls):
#         """Config tối ưu tốc độ - chỉ OCR + text cơ bản"""
#         return cls(
#             orientation=False,       # Bỏ qua orientation để tăng tốc
#             layout_detection=False,  # Bỏ qua layout detection
#             table_processing=False,
#             text_processing=True,
#             image_extraction=False,
#             xy_cut_sorting=False     # Dùng sort đơn giản
#         )
    
#     @classmethod
#     def table_focused_config(cls):
#         """Config tập trung vào bảng"""
#         return cls(
#             orientation=True,
#             layout_detection=True,   # Cần thiết để detect table
#             table_processing=True,
#             text_processing=True,    # Vẫn cần text xung quanh table
#             image_extraction=False,
#             xy_cut_sorting=True
#         )