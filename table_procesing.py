import cv2
import numpy as np
from paddleocr import TableCellsDetection, TextDetection
import os
import json
from datetime import datetime

# Import hàm OCR từ ocr.py
try:
    from ocr import run_ocr_on_image_input
except ImportError:
    print("Warning: Cannot import run_ocr_on_image_input from ocr.py")
    run_ocr_on_image_input = None

def preprocess_image(image_path, output_path=None):
    """
    Tiền xử lý ảnh để loại bỏ màu và chỉ giữ lại text màu đen/xám
    
    Args:
        image_path (str): Đường dẫn ảnh input
        output_path (str): Đường dẫn lưu ảnh đã xử lý (optional)
    
    Returns:
        numpy.ndarray: Ảnh đã được tiền xử lý
    """
    print(f"Preprocessing image: {image_path}")
    
    # Đọc ảnh màu
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image from {image_path}")
    
    # Tách các kênh màu
    b, g, r = cv2.split(img)
    
    # Tính độ lệch màu giữa các kênh → nếu lệch ít thì là màu xám
    color_diff = np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    
    # Tạo mask giữ lại pixel gần như xám (độ lệch thấp) và tối (giá trị nhỏ)
    gray_mask = (color_diff < 30) & (r < 200) & (g < 200) & (b < 200)
    
    # Tạo ảnh trắng toàn bộ
    output = np.ones_like(img) * 255
    
    # Chỗ nào là màu đen/xám thì giữ lại từ ảnh gốc
    output[gray_mask] = img[gray_mask]
    
    # Lưu ảnh kết quả nếu có đường dẫn
    if output_path:
        cv2.imwrite(output_path, output)
        print(f"Preprocessed image saved to: {output_path}")
    
    return output

def calculate_auto_tolerance(cell_boxes, tolerance_ratio=0.1):
    """
    Tính toán tolerance_x và tolerance_y tự động dựa trên kích thước cell nhỏ nhất
    
    Args:
        cell_boxes (list): Danh sách các cell boxes
        tolerance_ratio (float): Tỷ lệ tolerance so với kích thước cell nhỏ nhất
    
    Returns:
        tuple: (tolerance_x, tolerance_y)
    """
    if not cell_boxes:
        return 20, 20  # Default values
    
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
    
    # Tính tolerance dựa trên tỷ lệ của cell nhỏ nhất
    tolerance_x = max(5, int(min_width * tolerance_ratio))  # Minimum 5 pixels
    tolerance_y = max(5, int(min_height * tolerance_ratio))  # Minimum 5 pixels
    
    print(f"Auto-calculated tolerance: X={tolerance_x}, Y={tolerance_y}")
    print(f"Based on min cell size: width={min_width:.1f}, height={min_height:.1f}")
    
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

def polygon_to_bbox(polygon):
    """Chuyển polygon thành bounding box"""
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

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
    
    # Tạo visualization
    vis_image = image.copy()
    
    # Vẽ các cell gốc (màu đỏ)
    for cell in cells:
        x1, y1, x2, y2 = map(int, cell["coordinate"])
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Vẽ các cell mới (màu xanh lá)
    for cell in merged_new_cells:
        x1, y1, x2, y2 = map(int, cell["coordinate"])
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Vẽ outline của bảng
    cv2.rectangle(vis_image, (x1_table, y1_table), (x2_table, y2_table), (255, 0, 0), 2)
    
    return merged_new_cells, vis_image, table_bbox

def crop_and_ocr_cells(image, cells):
    """
    Crop từng cell và chạy OCR để lấy nội dung text
    
    Args:
        image: Ảnh gốc (numpy array)
        cells: Danh sách các cell với coordinate
    
    Returns:
        List[dict]: Danh sách cells với thêm trường 'text' chứa nội dung OCR
    """
    if run_ocr_on_image_input is None:
        print("Warning: OCR function not available. Skipping OCR processing.")
        return cells
    
    cells_with_text = []
    
    print(f"Processing OCR for {len(cells)} cells...")
    
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
        
        # Kiểm tra kích thước crop hợp lệ
        if cropped_cell.size == 0:
            print(f"Warning: Cell {i} has invalid size, skipping OCR")
            cell_with_text = cell.copy()
            cell_with_text["text"] = ""
            cell_with_text["cell_id"] = i
            cells_with_text.append(cell_with_text)
            continue
        
        try:
            # Chạy OCR trên ảnh crop
            print(f"Running OCR on cell {i}...")
            ocr_text = run_ocr_on_image_input(cropped_cell, save_output=False)
            
            # Làm sạch text (loại bỏ khoảng trắng thừa)
            ocr_text = ocr_text.strip()
            
        except Exception as e:
            print(f"Error running OCR on cell {i}: {str(e)}")
            ocr_text = ""
        
        # Thêm thông tin text vào cell
        cell_with_text = cell.copy()
        cell_with_text["text"] = ocr_text
        cell_with_text["cell_id"] = i
        cells_with_text.append(cell_with_text)
        
        print(f"Cell {i}: '{ocr_text}'")
    
    print(f"OCR processing completed for {len(cells_with_text)} cells")
    return cells_with_text

def convert_to_serializable(obj):
    """Chuyển đổi các kiểu dữ liệu NumPy thành kiểu dữ liệu Python chuẩn"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def create_grid_structure_json(final_cells, output_path):
    """Tạo cấu trúc lưới (grid) từ các cell có kèm theo text content"""
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
                        "text": cell.get("text", ""),  # Thêm nội dung text
                        "cell_id": cell.get("cell_id", -1)  # ID của cell
                    }
    
    # Tạo dữ liệu JSON
    json_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_rows": len(y_coords) - 1,
            "total_columns": len(x_coords) - 1,
            "total_cells": len(grid),
            "x_coordinates": [float(x) for x in x_coords],
            "y_coordinates": [float(y) for y in y_coords]
        },
        "grid": grid
    }
    
    # Convert to serializable format
    json_data = convert_to_serializable(json_data)
    
    # Lưu file JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    
    return output_path

def process_table_with_ocr(
    image_path,
    output_dir="./output",
    # Cell detection parameters
    cell_threshold=0.7,
    cell_batch_size=1,
    # Coordinate normalization parameters
    tolerance_x=None,  # Sẽ được tính tự động nếu None
    tolerance_y=None,  # Sẽ được tính tự động nếu None
    tolerance_ratio=0.1,  # Tỷ lệ để tính tolerance tự động
    # Preprocessing parameters
    enable_preprocessing=True,
    save_preprocessed=True
):
    """
    Hàm tổng hợp xử lý ảnh table với OCR và các tham số có thể tùy chỉnh
    
    Args:
        image_path (str): Đường dẫn ảnh input
        output_dir (str): Thư mục lưu kết quả
        cell_threshold (float): Ngưỡng confidence cho cell detection
        cell_batch_size (int): Batch size cho cell detection
        tolerance_x (int): Dung sai tọa độ X khi normalize (None = tự động)
        tolerance_y (int): Dung sai tọa độ Y khi normalize (None = tự động)
        tolerance_ratio (float): Tỷ lệ để tính tolerance tự động
        enable_preprocessing (bool): Có chạy tiền xử lý ảnh không
        save_preprocessed (bool): Có lưu ảnh đã tiền xử lý không
        
    Returns:
        tuple: (vis_output_path, json_output_path, preprocessed_image_path)
    """
    print(f"Processing image: {image_path}")
    print(f"Parameters:")
    print(f"  - Cell threshold: {cell_threshold}")
    print(f"  - Cell batch size: {cell_batch_size}")
    print(f"  - Enable preprocessing: {enable_preprocessing}")
    print(f"  - Auto tolerance ratio: {tolerance_ratio}")
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    # Tiền xử lý ảnh nếu được bật
    preprocessed_image_path = None
    if enable_preprocessing:
        preprocessed_image_path = os.path.join(output_dir, "preprocessed_image.jpg") if save_preprocessed else None
        processed_image = preprocess_image(image_path, preprocessed_image_path)
        
        # Tạo file tạm để xử lý
        temp_image_path = os.path.join(output_dir, "temp_preprocessed.jpg")
        cv2.imwrite(temp_image_path, processed_image)
        working_image_path = temp_image_path
        working_image = processed_image
    else:
        working_image = cv2.imread(image_path)
        working_image_path = image_path
    
    if working_image is None:
        raise ValueError(f"Cannot load image from {image_path}")
    
    # Khởi tạo model cell detection
    print("Initializing cell detection model...")
    cell_model = TableCellsDetection(model_name="RT-DETR-L_wired_table_cell_det")
    
    # Detect cells
    print("Detecting cells...")
    cell_output = cell_model.predict(working_image_path, threshold=cell_threshold, batch_size=cell_batch_size)
    original_cells = cell_output[0]["boxes"] if cell_output else []
    
    print(f"Detected {len(original_cells)} cells")
    
    # Tính toán tolerance tự động nếu không được chỉ định
    if tolerance_x is None or tolerance_y is None:
        auto_tolerance_x, auto_tolerance_y = calculate_auto_tolerance(original_cells, tolerance_ratio)
        if tolerance_x is None:
            tolerance_x = auto_tolerance_x
        if tolerance_y is None:
            tolerance_y = auto_tolerance_y
    
    print(f"  - Final tolerance X: {tolerance_x}")
    print(f"  - Final tolerance Y: {tolerance_y}")
    
    # Coordinate normalization
    print("Normalizing coordinates...")
    normalized_cells = normalize_coordinates(original_cells, tolerance_x, tolerance_y)
    
    # Tạo cells mới cho vùng chưa được che phủ
    print("Creating new cells for uncovered areas...")
    new_cells, vis_image, table_bbox = create_cells_for_non_covered_areas(working_image, normalized_cells)
    
    # Gộp tất cả cells
    all_cells = normalized_cells + new_cells
    print(f"Total cells before OCR: {len(all_cells)} (original: {len(normalized_cells)}, new: {len(new_cells)})")
    
    # Crop và OCR từng cell
    print("Cropping cells and running OCR...")
    final_cells_with_text = crop_and_ocr_cells(working_image, all_cells)
    
    # Lưu ảnh visualization
    vis_output_path = os.path.join(output_dir, "table_cells_visualization.jpg")
    cv2.imwrite(vis_output_path, vis_image)
    print(f"Visualization saved to: {vis_output_path}")
    
    # Tạo và lưu grid structure JSON (có text content)
    json_output_path = os.path.join(output_dir, "grid_structure_with_text.json")
    create_grid_structure_json(final_cells_with_text, json_output_path)
    print(f"Grid structure with text content saved to: {json_output_path}")
    
    # Dọn dẹp file tạm nếu có
    if enable_preprocessing and os.path.exists(temp_image_path):
        try:
            os.remove(temp_image_path)
        except:
            pass
    
    print("Processing completed!")
    return vis_output_path, json_output_path, preprocessed_image_path

# Hàm wrapper để sử dụng với tham số mặc định
def process_table_image(image_path, output_dir="./output"):
    """
    Hàm wrapper với tham số mặc định cho khả năng tương thích ngược
    """
    return process_table_with_ocr(image_path, output_dir)

# Sử dụng
if __name__ == "__main__":
    # Đường dẫn ảnh input
    image_path = "./cropped_boxes/table_0.jpg"
    output_dir = "./output"
    
    vis_path, json_path, preprocessed_path = process_table_with_ocr(
        image_path=image_path,
        output_dir=output_dir,
        cell_threshold=0.7,        # Ngưỡng confidence cell detection
        cell_batch_size=1,         # Batch size cell detection  
        tolerance_x=None,          # Tự động tính dựa trên cell nhỏ nhất
        tolerance_y=None,          # Tự động tính dựa trên cell nhỏ nhất
        tolerance_ratio=0.5,       # Tỷ lệ để tính tolerance (10% kích thước cell nhỏ nhất)
        enable_preprocessing=True, # Bật tiền xử lý ảnh
        save_preprocessed=True     # Lưu ảnh đã tiền xử lý
    )
    
    print(f"\nOutput files:")
    print(f"- Visualization: {vis_path}")
    print(f"- Grid structure: {json_path}")
    if preprocessed_path:
        print(f"- Preprocessed image: {preprocessed_path}")