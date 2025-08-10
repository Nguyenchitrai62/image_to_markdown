import os
from img_2_md import DocumentProcessor, process_single_document

test_image = "./anh_test/1.jpg"
if os.path.exists(test_image):
    result = process_single_document(test_image, "./output/")
    print(f"✓ Result: {result}")
else:
    print("✗ Test image not found")