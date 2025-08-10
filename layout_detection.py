import cv2
import numpy as np
from paddleocr import LayoutDetection
import paddle
import os
import json

def initialize_layout_detector():
    """Khởi tạo PaddleOCR LayoutDetection"""
    print("Paddle đang dùng thiết bị:", paddle.device.get_device())
    
    layout_model = LayoutDetection(
        model_name="PP-DocLayout-L"
    )
    
    return layout_model

def process_layout_boxes(boxes, image_shape):
    """Xử lý và chuẩn hóa các boxes từ layout detection"""
    processed_boxes = []
    
    for box_info in boxes:
        label = box_info["label"]
        coordinate = box_info["coordinate"]  # [x1, y1, x2, y2, x3, y3, x4, y4]
        
        # Lấy bounding box hình chữ nhật giới hạn
        x_coords = coordinate[::2]
        y_coords = coordinate[1::2]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # Đảm bảo tọa độ nằm trong giới hạn ảnh
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_shape[1], x_max)
        y_max = min(image_shape[0], y_max)
        
        box_data = {
            'label': label,
            'bbox': [x_min, y_min, x_max, y_max],
            'coordinate': coordinate,
            'width': x_max - x_min,
            'height': y_max - y_min
        }
        processed_boxes.append(box_data)
    
    return processed_boxes

def crop_layout_regions(image, processed_boxes):
    """Cắt các vùng theo layout boxes"""
    cropped_regions = []
    
    for i, box_data in enumerate(processed_boxes):
        x_min, y_min, x_max, y_max = box_data['bbox']
        
        # Cắt ảnh con
        if x_max > x_min and y_max > y_min:
            cropped = image[y_min:y_max, x_min:x_max]
            
            region_info = {
                'image': cropped,
                'label': box_data['label'],
                'bbox': box_data['bbox'],
                'index': i,
                'size': (box_data['width'], box_data['height'])
            }
            cropped_regions.append(region_info)
    
    return cropped_regions

def run_layout_detection(image_path, layout_model=None):
    """
    Hàm chính: Xử lý layout detection
    
    Args:
        image_path (str): Đường dẫn đến ảnh
        layout_model: Model layout detection (nếu None sẽ khởi tạo mới)
    
    Returns:
        tuple: (original_image, cropped_regions, layout_boxes)
            - original_image: Ảnh gốc
            - cropped_regions: List các vùng đã cắt với thông tin
            - layout_boxes: List các box với thông tin layout
    """
    if layout_model is None:
        layout_model = initialize_layout_detector()
    
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    print(f"Đang xử lý layout detection: {image_path}")
    
    # Dự đoán layout
    output = layout_model.predict(image_path, batch_size=1, layout_nms=True)
    
    all_regions = []
    all_boxes = []
    
    for idx, res in enumerate(output):
        boxes = res["boxes"]
        print(f"Tìm thấy {len(boxes)} vùng layout")
        
        # Xử lý boxes
        processed_boxes = process_layout_boxes(boxes, image.shape)
        all_boxes.extend(processed_boxes)
        
        # Cắt các vùng
        cropped_regions = crop_layout_regions(image, processed_boxes)
        all_regions.extend(cropped_regions)
        
        # In thông tin các vùng tìm được
        for i, region in enumerate(cropped_regions):
            print(f"Vùng {i+1}: {region['label']} - Kích thước: {region['size']}")
    
    return image, all_regions, all_boxes

def save_layout_results(original_image, cropped_regions, layout_boxes, image_path, output_dir="./output/"):
    """Lưu kết quả layout detection"""
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Tạo thư mục con cho ảnh cắt
    crop_dir = os.path.join(output_dir, f"{base_name}_cropped")
    os.makedirs(crop_dir, exist_ok=True)
    
    # Lưu các ảnh cắt
    saved_crops = []
    for region in cropped_regions:
        filename = f"{region['label']}_{region['index']}.jpg"
        crop_path = os.path.join(crop_dir, filename)
        
        if region['image'].size > 0:  # Đảm bảo ảnh không rỗng
            cv2.imwrite(crop_path, region['image'])
            saved_crops.append({
                'filename': filename,
                'path': crop_path,
                'label': region['label'],
                'bbox': region['bbox'],
                'size': region['size']
            })
            print(f"Đã lưu: {crop_path}")
    
    # Lưu thông tin layout boxes
    layout_result_path = os.path.join(output_dir, f"{base_name}_layout_result.json")
    result_data = {
        'image_info': {
            'path': image_path,
            'shape': original_image.shape,
            'total_regions': len(cropped_regions)
        },
        'layout_boxes': layout_boxes,
        'cropped_regions': saved_crops
    }
    
    with open(layout_result_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"Đã lưu: {layout_result_path}")
    
    return saved_crops

def get_regions_by_label(cropped_regions, target_label):
    """Lọc các vùng theo label cụ thể"""
    filtered_regions = [region for region in cropped_regions if region['label'] == target_label]
    return filtered_regions

def visualize_layout_boxes(image, layout_boxes, save_path=None):
    """Vẽ các boxes lên ảnh để visualization"""
    vis_image = image.copy()
    
    # Màu cho các loại label khác nhau
    colors = {
        'text':   (0, 100, 0),       # Dark Green (R:0, G:100, B:0)
        'title':  (0, 0, 139),       # Dark Red   (R:139, G:0, B:0)
        'figure': (139, 0, 0),       # Dark Blue  (R:0, G:0, B:139)
        'table':  (11, 134, 184),    # DarkGoldenRod (R:184, G:134, B:11)
        'list':   (211, 0, 148),     # DarkViolet (R:148, G:0, B:211)
        'header': (139, 139, 0),     # DarkCyan   (R:0, G:139, B:139)
        'footer': (64, 64, 64)       # Dark Gray
    }
    
    for box in layout_boxes:
        x_min, y_min, x_max, y_max = box['bbox']
        label = box['label']
        color = colors.get(label, (0, 0, 0))
        
        # Vẽ rectangle
        cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 3)
        
        # Vẽ label
        cv2.putText(vis_image, label, (x_min, y_min-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    if save_path:
        cv2.imwrite(save_path, vis_image)
        print(f"Đã lưu visualization: {save_path}")
    
    return vis_image

# CÁCH TEST:
# 1. Test cơ bản:
#    layout_model = initialize_layout_detector()
#    original_img, regions, boxes = run_layout_detection("./anh_test/1.jpg", layout_model)
#    save_layout_results(original_img, regions, boxes, "./anh_test/1.jpg")
#
# 2. Test nhiều ảnh:
#    layout_model = initialize_layout_detector()
#    for img_path in ["./anh_test/1.jpg", "./anh_test/2.jpg"]:
#        original_img, regions, boxes = run_layout_detection(img_path, layout_model)
#        save_layout_results(original_img, regions, boxes, img_path)
#
# 3. Lọc theo label:
#    text_regions = get_regions_by_label(regions, 'text')
#    table_regions = get_regions_by_label(regions, 'table')

if __name__ == "__main__":
    # Test đơn giản
    layout_model = initialize_layout_detector()
    image_path = "./anh_test/6.jpg"
    
    # Xử lý layout detection
    original_img, regions, boxes = run_layout_detection(image_path, layout_model)
    
    # In kết quả
    print(f"Kích thước ảnh gốc: {original_img.shape}")
    print(f"Số vùng tìm được: {len(regions)}")
    
    # In thông tin chi tiết các vùng
    label_counts = {}
    for region in regions:
        label = region['label']
        label_counts[label] = label_counts.get(label, 0) + 1
        
    print("Thống kê các vùng:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} vùng")
    
    # Lưu kết quả
    saved_crops = save_layout_results(original_img, regions, boxes, image_path)
    
    # Tạo visualization
    vis_image = visualize_layout_boxes(original_img, boxes, "./output/layout_visualization.jpg")
    
    # Ví dụ lọc theo label
    text_regions = get_regions_by_label(regions, 'text')
    print(f"Số vùng text: {len(text_regions)}")