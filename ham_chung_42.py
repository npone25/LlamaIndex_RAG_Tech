# Các hàm dùng chung cho thí nghiệm 4.1.2 và mục 4.2
# row_to_text, row_to_document, doc_bang_dien_thoai lấy từ chuyen_doi_san_pham.py

import os
import sys
from importlib.metadata import version
from pathlib import Path

from llama_index.core import Document
from llama_index.core.utils import get_tokenizer

from chuyen_doi_san_pham import doc_bang_dien_thoai, row_to_document, row_to_text

sys.stdout.reconfigure(encoding="utf-8")  # để in được tiếng Việt trên Windows

FOLDER = Path(__file__).parent
POLICY_FILE = FOLDER / "Data_for_exp_42" / "chinh_sach_bao_hanh_doi_tra.md"
PDF_FILE = FOLDER / "Data_for_exp" / "iphone-18-pro-max-info.pdf"

# Bộ đếm token mà SentenceSplitter dùng mặc định (tiktoken cl100k_base)
tokenizer = get_tokenizer()


def in_phien_ban():
    print("Python", sys.version.split()[0], "| llama-index-core", version("llama-index-core"))


def hide_path(text):
    # Bỏ phần đường dẫn của máy để log gọn và không lộ thông tin
    return str(text).replace(str(FOLDER) + os.sep, "")


def rut_gon(text, so_ky_tu=80):
    # Cắt ngắn và bỏ xuống dòng để mỗi dòng log vừa một ảnh chụp
    text = str(text).replace("\n", " ")
    if len(text) <= so_ky_tu:
        return text
    return text[:so_ky_tu] + "..."


def dem_token(text):
    return len(tokenizer(text))


def lay_document_dai_nhat(documents):
    dai_nhat = documents[0]
    for doc in documents:
        if len(doc.text) > len(dai_nhat.text):
            dai_nhat = doc
    return dai_nhat


def doc_chinh_sach():
    """Đọc văn bản chính sách bảo hành - đổi trả (văn bản mẫu) thành một Document."""
    text = POLICY_FILE.read_text(encoding="utf-8")
    return Document(
        text=text,
        doc_id="Data_for_exp_42/chinh_sach_bao_hanh_doi_tra.md",  # đường dẫn tương đối, dùng /
        metadata={"file_name": POLICY_FILE.name, "doc_type": "chinh_sach"},
    )
