import json
import os
from collections import defaultdict

def load_grid_structure(json_path):
    """
    Đọc file grid_structure.json
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def build_table_matrix(grid_data):
    """
    Xây dựng ma trận bảng từ grid data với nội dung text thực tế
    """
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
            
            # Lấy nội dung text từ OCR, nếu không có thì dùng placeholder
            text_content = cell_info.get('text', '').strip()
            if not text_content:
                text_content = f"R{row+1}C{col+1}"  # Fallback content
            
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
    """
    Escape HTML characters trong text content
    """
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

def generate_html_table_simple(table_matrix, total_rows, total_columns):
    """
    Tạo HTML table đơn giản với nội dung text thực tế
    """
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
                content = cell['content']
                if content.startswith('R') and content[1:].split('C')[0].isdigit():
                    content = ""
                else:
                    content = escape_html(content)
                
                html_lines.append(f'    <td{attrs_str}>{content}</td>')
        
        html_lines.append('  </tr>')
    
    html_lines.append('</table>')
    
    return '\n'.join(html_lines)

def generate_markdown_table_simple(table_matrix, total_rows, total_columns, output_path=None):
    """
    Tạo file Markdown chỉ chứa HTML table đơn giản
    """
    if not table_matrix:
        return None
    
    # Chỉ tạo HTML table đơn giản
    html_table = generate_html_table_simple(table_matrix, total_rows, total_columns)
    
    # Lưu file nếu có đường dẫn
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_table)
        print(f"Markdown table saved to: {output_path}")
    
    return html_table

def convert_grid_to_tables(grid_json_path, output_dir="./output", custom_content=None):
    """
    Chuyển đổi grid_structure.json thành file Markdown chỉ chứa HTML table
    
    Args:
        grid_json_path: Đường dẫn đến file grid_structure_with_text.json
        output_dir: Thư mục output
        custom_content: Dict mapping tùy chỉnh content cho cells
    
    Returns:
        str: html table content
    """
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 Converting grid structure to markdown table...")
    print(f"📁 Input: {grid_json_path}")
    print(f"📁 Output: {output_dir}")
    
    # Đọc grid structure
    grid_data = load_grid_structure(grid_json_path)
    if not grid_data:
        print("❌ Failed to load grid structure")
        return None
    
    # Xây dựng ma trận bảng
    table_matrix, dimensions = build_table_matrix(grid_data)
    if not table_matrix or not dimensions:
        print("❌ Failed to build table matrix")
        return None
    
    total_rows, total_columns = dimensions
    print(f"📊 Table dimensions: {total_rows} rows × {total_columns} columns")
    
    # Tùy chỉnh content nếu cần
    if custom_content:
        for row in range(total_rows):
            for col in range(total_columns):
                cell = table_matrix[row][col]
                if cell and f"R{row+1}C{col+1}" in custom_content:
                    cell['content'] = custom_content[f"R{row+1}C{col+1}"]
    
    # Tạo Markdown table (chỉ chứa HTML table)
    md_output_path = os.path.join(output_dir, "table.md")
    markdown_content = generate_markdown_table_simple(table_matrix, total_rows, total_columns, md_output_path)
    
    # Tạo thống kê
    detected_cells = sum(1 for row in table_matrix for cell in row if cell and cell['type'] == 'detected')
    created_cells = sum(1 for row in table_matrix for cell in row if cell and cell['type'] == 'created')
    total_cells = detected_cells + created_cells
    
    # Thống kê nội dung
    cells_with_content = sum(1 for row in table_matrix for cell in row 
                           if cell and not (cell['content'].startswith('R') and 
                                          cell['content'][1:].split('C')[0].isdigit()))
    
    print(f"\n✅ Conversion completed!")
    print(f"📊 Statistics:")
    print(f"   • Total cells: {total_cells}")
    print(f"   • Detected cells: {detected_cells}")
    print(f"   • Created cells: {created_cells}")
    print(f"   • Cells with OCR content: {cells_with_content}")
    print(f"📁 Output file: {md_output_path}")
    
    return markdown_content

# Hàm tiện ích để chạy conversion
def run_grid_conversion(grid_json_path="./output/grid_structure_with_text.json", 
                       output_dir="./output"):
    """
    Chạy conversion với đường dẫn mặc định cho file có OCR content
    """
    return convert_grid_to_tables(
        grid_json_path=grid_json_path,
        output_dir=output_dir
    )

if __name__ == "__main__":
    # Chạy conversion với OCR content
    try:
        print("🚀 Starting Grid to Table Conversion...")
        print("=" * 60)
        
        # Sử dụng file có OCR content
        markdown_content = run_grid_conversion(
            grid_json_path="./output/grid_structure_with_text.json"
        )
        
        if markdown_content:
            print("=" * 60)
            print("✅ Conversion completed successfully!")
            print("\n🔍 Check the table.md file for the HTML table!")
        else:
            print("❌ Conversion failed!")
            
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()