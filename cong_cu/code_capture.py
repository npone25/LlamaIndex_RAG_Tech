from pathlib import Path
from pygments import highlight
from pygments.lexers import PythonLexer, TextLexer
from pygments.formatters import ImageFormatter
from PIL import Image, ImageDraw, ImageFont
import io
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUT_DIR = Path("docs/anh_bao_cao")  # chạy ở thư mục gốc: python cong_cu/code_capture.py
OUT_DIR.mkdir(exist_ok=True)

def tao_anh_codesnap(file_path: Path, is_log: bool = False):
    text = file_path.read_text(encoding="utf-8", errors="replace").expandtabs(4)
    if not text.strip():
        return

    lexer = TextLexer() if is_log else PythonLexer()
    
    # 1. Tô màu cú pháp code bằng Pygments (Theme Monokai/Dracula)
    formatter = ImageFormatter(
        font_name="Consolas",
        font_size=16,
        line_numbers=not is_log,       # Code thì bật số dòng, Log thì tắt
        style="monokai",
        line_pad=4,
        image_pad=20,
        line_number_bg="#272822",
        line_number_fg="#8f908a",
    )
    raw_code_bytes = highlight(text, lexer, formatter)
    code_img = Image.open(io.BytesIO(raw_code_bytes)).convert("RGBA")

    # 2. Vẽ khung cửa sổ macOS (thanh tiêu đề + 3 nút tròn + tên file)
    header_height = 42
    padding = 24
    total_w = code_img.width + padding * 2
    total_h = code_img.height + header_height + padding * 2

    canvas = Image.new("RGBA", (total_w, total_h), (30, 32, 48, 255)) # Viền ngoài
    draw = ImageDraw.Draw(canvas)

    # Khung cửa sổ code bo góc
    win_box = [padding, padding, total_w - padding, total_h - padding]
    draw.rounded_rectangle(win_box, radius=10, fill=(39, 40, 34, 255))

    # Vẽ 3 nút tròn Đỏ - Vàng - Xanh
    cy = padding + header_height // 2
    colors = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]
    for i, col in enumerate(colors):
        cx = padding + 20 + i * 22
        draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=col)

    # Ghi tên file lên thanh tiêu đề
    try:
        font = ImageFont.truetype("consola.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((padding + 95, cy - 7), file_path.name, fill=(200, 200, 200), font=font)

    # Ghép phần code vào dưới thanh tiêu đề
    canvas.paste(code_img, (padding, padding + header_height))

    # Lưu file ảnh PNG
    prefix = "log_" if is_log else "code_"
    out_path = OUT_DIR / f"{prefix}{file_path.stem}.png"
    canvas.save(out_path)
    try:
        print(f"[✓] Đã chụp: {out_path}")
    except Exception:
        print(f"[OK] Da chup: {out_path}")

if __name__ == "__main__":
    # Chụp toàn bộ các file .py trong thư mục chính
    skip_files = {"code_capture.py", "chup_anh_hang_loat.py"}
    for py_file in sorted(list(Path("thi_nghiem").glob("*.py")) + list(Path("loi_rag").glob("*.py"))):
        if py_file.name not in skip_files:
            tao_anh_codesnap(py_file, is_log=False)

    # Chụp toàn bộ các file .txt trong thư mục logs/
    log_dir = Path("logs")
    if log_dir.exists():
        for log_file in sorted(log_dir.glob("*.txt")):
            tao_anh_codesnap(log_file, is_log=True)