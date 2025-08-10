import cv2
import numpy as np
from paddleocr import DocImgOrientationClassification
import paddle
import os
import json

def initialize_orientation_classifier():
    """Khởi tạo DocImgOrientationClassification"""
    print("Paddle đang dùng thiết bị:", paddle.device.get_device())
    
    classifier = DocImgOrientationClassification(
        model_name="PP-LCNet_x1_0_doc_ori"
    )
    
    return classifier

def rotate_image_by_angle(image, angle):
    """
    Xoay ảnh theo góc được xác định
    
    Args:
        image: numpy array của ảnh
        angle: góc xoay (0, 90, 180, 270)
    
    Returns:
        numpy array: ảnh đã được xoay
    """
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Nếu góc không phải 0, 90, 180, 270 thì dùng affine transform
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                     flags=cv2.INTER_CUBIC, 
                                     borderMode=cv2.BORDER_REPLICATE)
        return rotated_image

def get_rotation_angle_from_label(label_name):
    """
    Chuyển đổi label name thành góc xoay cần thiết để sửa orientation
    
    Args:
        label_name: tên label từ model ("0", "90", "180", "270")
    
    Returns:
        int: góc cần xoay để sửa orientation (ngược lại với góc hiện tại)
    """
    # Mapping từ orientation hiện tại sang góc cần xoay để sửa
    correction_angles = {
        "0": 0,      # Ảnh đã đúng hướng
        "90": 270,   # Ảnh xoay 90° clockwise, cần xoay 270° để sửa
        "180": 180,  # Ảnh xoay 180°, cần xoay 180° để sửa
        "270": 90    # Ảnh xoay 270° clockwise, cần xoay 90° để sửa
    }
    
    return correction_angles.get(str(label_name), 0)

def run_orientation_correction(input_data, orientation_model=None):
    """
    Hàm chính: Phát hiện orientation và xoay ảnh về đúng hướng
    
    Args:
        input_data: có thể là đường dẫn ảnh (str) hoặc numpy array
        orientation_model: model đã được khởi tạo (optional)
    
    Returns:
        tuple: (corrected_image, orientation_info)
            - corrected_image: numpy array của ảnh đã xoay đúng hướng
            - orientation_info: dict chứa thông tin về orientation
    """
    if orientation_model is None:
        orientation_model = initialize_orientation_classifier()
    
    # Xử lý input
    if isinstance(input_data, str):
        # Input là đường dẫn ảnh
        image_path = input_data
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        print(f"Đang xử lý ảnh: {image_path}")
    elif isinstance(input_data, np.ndarray):
        # Input là numpy array
        img = input_data.copy()
        image_path = None
        print("Đang xử lý ảnh từ numpy array")
    else:
        raise ValueError("Input phải là đường dẫn ảnh (str) hoặc numpy array")
    
    # Lưu ảnh tạm nếu input là numpy array
    temp_path = None
    if image_path is None:
        temp_path = "./temp_orientation_input.jpg"
        cv2.imwrite(temp_path, img)
        image_path = temp_path
    
    try:
        # Chạy orientation detection
        output = orientation_model.predict(image_path, batch_size=1)
        
        orientation_info = {
            'original_orientation': None,
            'confidence': None,
            'correction_angle': 0,
            'corrected': False
        }
        
        corrected_image = img.copy()
        
        for res in output:
            label_names = res["label_names"]
            print(f"Orientation được phát hiện: {label_names}")
            
            # Lấy thông tin chi tiết
            if hasattr(res, 'scores') and res.scores is not None:
                confidence = float(max(res.scores))
            else:
                confidence = 1.0
            
            # Tính góc cần xoay để sửa
            correction_angle = get_rotation_angle_from_label(label_names[0])
            
            # Xoay ảnh nếu cần
            if correction_angle != 0:
                corrected_image = rotate_image_by_angle(img, correction_angle)
                corrected = True
                print(f"Đã xoay ảnh {correction_angle}° để sửa orientation")
            else:
                corrected = False
                print("Ảnh đã ở đúng orientation")
            
            # Cập nhật thông tin
            orientation_info.update({
                'original_orientation': label_names[0],
                'confidence': confidence,
                'correction_angle': correction_angle,
                'corrected': corrected
            })
            
            break  # Chỉ xử lý result đầu tiên
    
    finally:
        # Xóa file tạm nếu có
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    
    return corrected_image, orientation_info

def save_corrected_results(corrected_image, orientation_info, original_path=None, output_dir="./output/"):
    """Lưu kết quả (chỉ để test)"""
    os.makedirs(output_dir, exist_ok=True)
    
    if original_path:
        base_name = os.path.splitext(os.path.basename(original_path))[0]
    else:
        base_name = "corrected_image"
    
    # Lưu ảnh đã xoay
    corrected_path = os.path.join(output_dir, f"{base_name}_corrected.jpg")
    cv2.imwrite(corrected_path, corrected_image)
    
    # Lưu thông tin orientation
    info_path = os.path.join(output_dir, f"{base_name}_orientation_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(orientation_info, f, ensure_ascii=False, indent=2)
    
    print(f"Đã lưu ảnh đã sửa: {corrected_path}")
    print(f"Đã lưu thông tin orientation: {info_path}")

# CÁCH TEST:
# 1. Test với đường dẫn ảnh:
#    orientation_model = initialize_orientation_classifier()
#    corrected_img, info = run_orientation_correction("./anh_test/12.jpg", orientation_model)
#    save_corrected_results(corrected_img, info, "./anh_test/12.jpg")
#
# 2. Test với numpy array:
#    orientation_model = initialize_orientation_classifier()
#    img = cv2.imread("./anh_test/12.jpg")
#    corrected_img, info = run_orientation_correction(img, orientation_model)
#    save_corrected_results(corrected_img, info)
#
# 3. Test nhiều ảnh:
#    orientation_model = initialize_orientation_classifier()
#    for img_path in ["./anh_test/12.jpg", "./anh_test/13.jpg"]:
#        corrected_img, info = run_orientation_correction(img_path, orientation_model)
#        save_corrected_results(corrected_img, info, img_path)

if __name__ == "__main__":
    # Test đơn giản
    orientation_model = initialize_orientation_classifier()
    image_path = "./anh_test/12.jpg"
    
    # Xử lý ảnh
    corrected_img, orientation_info = run_orientation_correction(image_path, orientation_model)
    
    # In kết quả
    print(f"Kích thước ảnh: {corrected_img.shape}")
    print(f"Orientation gốc: {orientation_info['original_orientation']}")
    print(f"Confidence: {orientation_info['confidence']:.3f}")
    print(f"Góc đã xoay: {orientation_info['correction_angle']}°")
    print(f"Đã sửa orientation: {orientation_info['corrected']}")
    
    # Lưu kết quả để test
    save_corrected_results(corrected_img, orientation_info, image_path)