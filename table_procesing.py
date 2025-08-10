import cv2
import numpy as np
from paddleocr import TableCellsDetection
from collections import defaultdict
from det_rec_preprocess import get_text


def initialize_cell_detector():
    """Khởi tạo PaddleOCR TableCellsDetection"""
    print("Đang khởi tạo Cell Detection model...")
    
    cell_model = TableCellsDetection(model_name="RT-DETR-L_wired_table_cell_det")
    
    print("Cell Detection model đã được khởi tạo thành công!")
    return cell_model

def calculate_auto_tolerance(cell_boxes, tolerance_ratio=0.5):
    """
    Tính toán tolerance_x và tolerance_y tự động dựa trên kích thước cell nhỏ nhất
    """
    if not cell_boxes:
        return 20, 20
    
    min_width = float('inf')
    min_height = float('inf')
    
    for cell in cell_boxes:
        x1, y1, x2, y2 = cell["coordinate"]
        width = x2 - x1
        height = y2 - y1
        
        if width > 0:
            min_width = min(min_width, width)
        if height > 0:
            min_height = min(min_height, height)
    
    tolerance_x = max(5, int(min_width * tolerance_ratio))
    tolerance_y = max(5, int(min_height * tolerance_ratio))
    
    return tolerance_x, tolerance_y

def calculate_iou(box1, box2):
    """Tính IoU (Intersection over Union) giữa 2 bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def merge_boxes(box1, box2):
    """Gộp 2 boxes thành 1 box lớn bao trọn cả 2"""
    return [
        min(box1[0], box2[0]),
        min(box1[1], box2[1]), 
        max(box1[2], box2[2]),
        max(box1[3], box2[3])
    ]

def normalize_coordinates(cell_boxes, tolerance_x=10, tolerance_y=10):
    """Chuẩn hóa tọa độ các cell để tạo grid đều đặn"""
    if not cell_boxes:
        return cell_boxes
    
    x_coords = []
    y_coords = []
    
    for cell in cell_boxes:
        x1, y1, x2, y2 = cell["coordinate"]
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    normalized_x = group_similar_coordinates(x_coords, tolerance_x)
    normalized_y = group_similar_coordinates(y_coords, tolerance_y)
    
    normalized_cells = []
    for cell in cell_boxes:
        x1, y1, x2, y2 = cell["coordinate"]
        
        norm_x1 = find_nearest_normalized_coord(x1, normalized_x)
        norm_y1 = find_nearest_normalized_coord(y1, normalized_y)
        norm_x2 = find_nearest_normalized_coord(x2, normalized_x)
        norm_y2 = find_nearest_normalized_coord(y2, normalized_y)
        
        norm_x1, norm_x2 = min(norm_x1, norm_x2), max(norm_x1, norm_x2)
        norm_y1, norm_y2 = min(norm_y1, norm_y2), max(norm_y1, norm_y2)
        
        normalized_cell = cell.copy()
        normalized_cell["coordinate"] = [norm_x1, norm_y1, norm_x2, norm_y2]
        normalized_cells.append(normalized_cell)
    
    return normalized_cells

def group_similar_coordinates(coords, tolerance):
    """Nhóm các tọa độ gần nhau và trả về giá trị đại diện"""
    unique_coords = sorted(set(coords))
    groups = []
    
    for coord in unique_coords:
        added_to_group = False
        for group in groups:
            if abs(coord - group[0]) <= tolerance:
                group.append(coord)
                added_to_group = True
                break
        
        if not added_to_group:
            groups.append([coord])
    
    normalized_mapping = {}
    for group in groups:
        normalized_coord = sum(group) / len(group)
        for coord in group:
            normalized_mapping[coord] = normalized_coord
    
    return normalized_mapping

def find_nearest_normalized_coord(coord, normalized_mapping):
    """Tìm tọa độ chuẩn hóa gần nhất"""
    if coord in normalized_mapping:
        return normalized_mapping[coord]
    
    min_distance = float('inf')
    nearest_coord = coord
    
    for original_coord, normalized_coord in normalized_mapping.items():
        distance = abs(coord - original_coord)
        if distance < min_distance:
            min_distance = distance
            nearest_coord = normalized_coord
    
    return nearest_coord

def create_cells_for_non_covered_areas(image, cells):
    """Tạo các cell mới để che phủ các vùng chưa được che phủ trong bảng"""
    if not cells:
        return [], None, None
    
    # Tính bounding box của toàn bộ bảng
    x_coords = []
    y_coords = []
    for cell in cells:
        x1, y1, x2, y2 = cell["coordinate"]
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])
    
    table_bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    x1_table, y1_table, x2_table, y2_table = map(int, table_bbox)
    
    # Tạo mask nhị phân
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Vẽ vùng bảng lên mask với giá trị 255
    cv2.rectangle(mask, (x1_table, y1_table), (x2_table, y2_table), 255, -1)
    
    # Vẽ các cell lên mask với giá trị 0
    for cell in cells:
        cx1, cy1, cx2, cy2 = map(int, cell["coordinate"])
        cv2.rectangle(mask, (cx1, cy1), (cx2, cy2), 0, -1)
    
    # Lấy tọa độ x, y từ các cell hiện tại
    x_coords = sorted(set(x_coords))
    y_coords = sorted(set(y_coords))
    
    # Tạo các cell mới cho vùng chưa được che phủ
    new_cells = []
    
    for i in range(len(y_coords) - 1):
        for j in range(len(x_coords) - 1):
            x1 = x_coords[j]
            y1 = y_coords[i]
            x2 = x_coords[j + 1]
            y2 = y_coords[i + 1]
            
            # Kiểm tra nếu ô nằm trong bảng
            if x1 >= x1_table and x2 <= x2_table and y1 >= y1_table and y2 <= y2_table:
                # Kiểm tra nếu ô có vùng chưa được che phủ
                cell_region = mask[int(y1):int(y2), int(x1):int(x2)]
                if np.any(cell_region == 255):
                    # Kiểm tra xem ô có giao với cell hiện tại không
                    overlaps = False
                    for cell in cells:
                        if calculate_iou([x1, y1, x2, y2], cell["coordinate"]) > 0:
                            overlaps = True
                            break
                    if not overlaps:
                        new_cells.append({"coordinate": [x1, y1, x2, y2], "confidence": 0.0})
    
    # Gộp các cell mới theo hàng và cột
    merged_new_cells = []
    used_indices = set()
    
    for i, cell1 in enumerate(new_cells):
        if i in used_indices:
            continue
        current_bbox = cell1["coordinate"]
        merged_indices = {i}
        
        # Gộp theo hàng
        changed = True
        while changed:
            changed = False
            for j, cell2 in enumerate(new_cells):
                if j in used_indices or j == i:
                    continue
                x1_1, y1_1, x2_1, y2_1 = current_bbox
                x1_2, y1_2, x2_2, y2_2 = cell2["coordinate"]
                
                if abs(y1_1 - y1_2) < 1e-6 and abs(y2_1 - y2_2) < 1e-6 and (abs(x2_1 - x1_2) < 1e-6 or abs(x2_2 - x1_1) < 1e-6):
                    current_bbox = merge_boxes(current_bbox, cell2["coordinate"])
                    merged_indices.add(j)
                    used_indices.add(j)
                    changed = True
        
        merged_new_cells.append({"coordinate": current_bbox, "confidence": 0.0})
        used_indices.add(i)
    
    return merged_new_cells

def crop_and_ocr_cells(image, cells):
    """Crop từng cell và chạy OCR để lấy nội dung text"""
    if get_text is None:
        return cells
    
    cells_with_text = []
    
    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = map(int, cell["coordinate"])
        
        # Đảm bảo tọa độ hợp lệ
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Crop cell từ ảnh gốc
        cropped_cell = image[y1:y2, x1:x2]
        
        if cropped_cell.size == 0:
            cell_with_text = cell.copy()
            cell_with_text["text"] = ""
            cell_with_text["cell_id"] = i
            cells_with_text.append(cell_with_text)
            continue
        
        try:
            ocr_text = get_text(cropped_cell, save_output=False)
            ocr_text = ocr_text.strip()
        except Exception as e:
            ocr_text = ""
        
        cell_with_text = cell.copy()
        cell_with_text["text"] = ocr_text
        cell_with_text["cell_id"] = i
        cells_with_text.append(cell_with_text)
    
    return cells_with_text

def build_grid_data(final_cells):
    """Xây dựng cấu trúc grid data từ các cell"""
    if not final_cells:
        return None
    
    # Lấy tất cả tọa độ x và y để tạo lưới
    x_coords = set()
    y_coords = set()
    
    for cell in final_cells:
        x1, y1, x2, y2 = cell["coordinate"]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        x_coords.update([x1, x2])
        y_coords.update([y1, y2])
    
    x_coords = sorted(x_coords)
    y_coords = sorted(y_coords)
    
    # Tạo lưới
    grid = {}
    
    for cell in final_cells:
        x1, y1, x2, y2 = cell["coordinate"]
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        
        # Tìm vị trí hàng và cột
        start_col = x_coords.index(x1)
        end_col = x_coords.index(x2)
        start_row = y_coords.index(y1)
        end_row = y_coords.index(y2)
        
        # Tạo key cho cell
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                grid_key = f"row_{row}_col_{col}"
                
                if grid_key not in grid:
                    grid[grid_key] = {
                        "row": row,
                        "column": col,
                        "coordinate": [x1, y1, x2, y2],
                        "type": "detected" if float(cell.get("confidence", 0)) > 0 else "created",
                        "confidence": float(cell.get("confidence", 0.0)),
                        "span_rows": end_row - start_row,
                        "span_cols": end_col - start_col,
                        "is_merged": (end_row - start_row > 1) or (end_col - start_col > 1),
                        "text": cell.get("text", ""),
                        "cell_id": cell.get("cell_id", -1)
                    }
    
    return {
        "metadata": {
            "total_rows": len(y_coords) - 1,
            "total_columns": len(x_coords) - 1,
            "total_cells": len(grid)
        },
        "grid": grid
    }

def build_table_matrix(grid_data):
    """Xây dựng ma trận bảng từ grid data"""
    if not grid_data or 'grid' not in grid_data:
        return None, None
    
    metadata = grid_data.get('metadata', {})
    total_rows = metadata.get('total_rows', 0)
    total_columns = metadata.get('total_columns', 0)
    grid = grid_data['grid']
    
    # Tạo ma trận để theo dõi các cell đã được xử lý
    processed = [[False for _ in range(total_columns)] for _ in range(total_rows)]
    
    # Tạo ma trận để lưu thông tin cell
    table_matrix = [[None for _ in range(total_columns)] for _ in range(total_rows)]
    
    # Sắp xếp grid theo row và column
    sorted_grid_items = []
    for key, cell_info in grid.items():
        sorted_grid_items.append((cell_info['row'], cell_info['column'], key, cell_info))
    sorted_grid_items.sort()
    
    # Điền thông tin vào ma trận
    for row, col, key, cell_info in sorted_grid_items:
        if not processed[row][col]:
            span_rows = cell_info.get('span_rows', 1)
            span_cols = cell_info.get('span_cols', 1)
            cell_type = cell_info.get('type', 'unknown')
            
            # Đánh dấu các ô đã được xử lý
            for r in range(row, min(row + span_rows, total_rows)):
                for c in range(col, min(col + span_cols, total_columns)):
                    processed[r][c] = True
            
            # Lấy nội dung text từ OCR
            text_content = cell_info.get('text', '').strip()
            if not text_content:
                text_content = ""
            
            # Lưu thông tin cell
            table_matrix[row][col] = {
                'content': text_content,
                'rowspan': span_rows,
                'colspan': span_cols,
                'type': cell_type,
                'coordinate': cell_info.get('coordinate', []),
                'confidence': cell_info.get('confidence', 0),
                'cell_id': cell_info.get('cell_id', -1)
            }
    
    return table_matrix, (total_rows, total_columns)

def escape_html(text):
    """Escape HTML characters trong text content"""
    if not isinstance(text, str):
        text = str(text)
    
    html_escape_table = {
        "&": "&amp;",
        '"': "&quot;",
        "'": "&#x27;",
        ">": "&gt;",
        "<": "&lt;",
        "\n": "<br>",
        "\t": "&nbsp;&nbsp;&nbsp;&nbsp;"
    }
    
    for char, escaped in html_escape_table.items():
        text = text.replace(char, escaped)
    
    return text

def generate_html_table(table_matrix, total_rows, total_columns):
    """Tạo HTML table với nội dung text thực tế"""
    html_lines = []
    html_lines.append('<table border="1">')
    
    for row in range(total_rows):
        html_lines.append('  <tr>')
        
        for col in range(total_columns):
            cell = table_matrix[row][col]
            if cell is not None:
                td_attrs = []
                
                if cell['rowspan'] > 1:
                    td_attrs.append(f'rowspan="{cell["rowspan"]}"')
                
                if cell['colspan'] > 1:
                    td_attrs.append(f'colspan="{cell["colspan"]}"')
                
                attrs_str = ' ' + ' '.join(td_attrs) if td_attrs else ''
                
                # Sử dụng nội dung text thực tế
                content = escape_html(cell['content'])
                
                html_lines.append(f'    <td{attrs_str}>{content}</td>')
        
        html_lines.append('  </tr>')
    
    html_lines.append('</table>')
    
    return '\n'.join(html_lines)

def table_image_to_markdown(image_input, cell_model=None, cell_threshold=0.7, tolerance_ratio=0.5):
    """
    Chuyển đổi ảnh table thành markdown string
    
    Args:
        image_input: Đường dẫn ảnh hoặc numpy array
        cell_model: Model cell detection đã được khởi tạo (nếu None sẽ báo lỗi)
        cell_threshold: Ngưỡng confidence cho cell detection
        tolerance_ratio: Tỷ lệ để tính tolerance tự động
        
    Returns:
        str: Nội dung markdown (HTML table)
    """
    
    # Kiểm tra cell_model
    if cell_model is None:
        raise ValueError("cell_model không được để None. Hãy khởi tạo cell_model trước khi gọi hàm này.")
    
    # Đọc ảnh
    if isinstance(image_input, str):
        # Nếu là đường dẫn file
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Cannot load image from {image_input}")
        working_image_path = image_input
    elif isinstance(image_input, np.ndarray):
        # Nếu là numpy array
        image = image_input.copy()
        # Tạo file tạm để dùng với PaddleOCR
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, image)
            working_image_path = tmp_file.name
    else:
        raise ValueError("image_input must be either a file path or numpy array")
    
    try:
        # Sử dụng cell_model đã được truyền vào
        cell_output = cell_model.predict(working_image_path, threshold=cell_threshold, batch_size=1)
        original_cells = cell_output[0]["boxes"] if cell_output else []
        
        if not original_cells:
            return "<table border=\"1\"><tr><td>No table detected</td></tr></table>"
        
        # Tính toán tolerance tự động
        tolerance_x, tolerance_y = calculate_auto_tolerance(original_cells, tolerance_ratio)
        
        # Coordinate normalization
        normalized_cells = normalize_coordinates(original_cells, tolerance_x, tolerance_y)
        
        # Tạo cells mới cho vùng chưa được che phủ
        new_cells = create_cells_for_non_covered_areas(image, normalized_cells)
        
        # Gộp tất cả cells
        all_cells = normalized_cells + new_cells
        
        # Crop và OCR từng cell
        final_cells_with_text = crop_and_ocr_cells(image, all_cells)
        
        # Xây dựng grid data
        grid_data = build_grid_data(final_cells_with_text)
        
        if not grid_data:
            return "<table border=\"1\"><tr><td>Failed to process table</td></tr></table>"
        
        # Xây dựng ma trận bảng
        table_matrix, dimensions = build_table_matrix(grid_data)
        
        if not table_matrix or not dimensions:
            return "<table border=\"1\"><tr><td>Failed to build table matrix</td></tr></table>"
        
        total_rows, total_columns = dimensions
        
        # Tạo HTML table
        markdown_content = generate_html_table(table_matrix, total_rows, total_columns)
        
        return markdown_content
    
    finally:
        # Dọn dẹp file tạm nếu có
        if isinstance(image_input, np.ndarray) and os.path.exists(working_image_path):
            try:
                os.unlink(working_image_path)
            except:
                pass

# Alias cho function name ngắn hơn
def process_table_image(image_input, cell_model=None, **kwargs):
    """Alias cho table_image_to_markdown"""
    return table_image_to_markdown(image_input, cell_model, **kwargs)

if __name__ == "__main__":
    # Test với ảnh mẫu
    image_path = "./cropped_boxes/table_0.jpg"
    
    try:
        # Khởi tạo cell model một lần
        cell_model = initialize_cell_detector()
        
        # Sử dụng model để xử lý
        markdown_result = table_image_to_markdown(image_path, cell_model)
        print("Generated Markdown Table:")
        print("=" * 50)
        print(markdown_result)
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()