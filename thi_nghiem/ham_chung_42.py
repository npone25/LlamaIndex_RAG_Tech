# Các hàm dùng chung cho thí nghiệm 4.1.2 và mục 4.2
# row_to_text, row_to_document, doc_bang_dien_thoai lấy từ chuyen_doi_san_pham.py

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import os
import sys
from importlib.metadata import version
from pathlib import Path

from llama_index.core import Document
from llama_index.core.utils import get_tokenizer

from chuyen_doi_san_pham import load_phones, row_to_document, row_to_text

sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows

FOLDER = Path(__file__).parent
POLICY_FILE = FOLDER / "Data_for_exp_42" / "chinh_sach_bao_hanh_doi_tra.md"
PDF_FILE = FOLDER / "Data_for_exp" / "iphone-18-pro-max-info.pdf"

# Bộ đếm token mà SentenceSplitter dùng mặc định (tiktoken cl100k_base)
tokenizer = get_tokenizer()


def print_version():
    print("Python", sys.version.split()[0], "| llama-index-core", version("llama-index-core"))


def hide_path(text):
    # Bỏ phần đường dẫn của máy để log gọn và không lộ thông tin
    return str(text).replace(str(FOLDER) + os.sep, "")


def shorten(text, char_count=80):
    # Cắt ngắn và bỏ xuống dòng để mỗi dòng log vừa một ảnh chụp
    text = str(text).replace("\n", " ")
    if len(text) <= char_count:
        return text
    return text[:char_count] + "..."


def count_tokens(text):
    return len(tokenizer(text))


def get_longest_document(documents):
    longest = documents[0]
    for doc in documents:
        if len(doc.text) > len(longest.text):
            longest = doc
    return longest


def load_policies():
    """Đọc văn bản chính sách bảo hành - đổi trả (văn bản mẫu) thành một Document."""
    text = POLICY_FILE.read_text(encoding="utf-8")
    return Document(
        text=text,
        doc_id="Data_for_exp_42/chinh_sach_bao_hanh_doi_tra.md",  # đường dẫn tương đối, dùng /
        metadata={"file_name": POLICY_FILE.name, "doc_type": "chinh_sach"},
    )
