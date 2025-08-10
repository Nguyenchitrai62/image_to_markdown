import os
from img_2_md import DocumentProcessor

def test():
    """Test nhiều ảnh"""
    test_images = [
        "./anh_test/1.jpg", 
        "./anh_test/2.jpg", 
        "./anh_test/3.jpg"
        ]
    
    existing = [img for img in test_images if os.path.exists(img)]
    
    if existing:
        processor = DocumentProcessor()
        results = processor.process_multiple_documents(existing, "./output/")
        print(f"✓ Processed {len(results)} files")
    else:
        print("✗ No test images found")

if __name__ == "__main__":
    test()