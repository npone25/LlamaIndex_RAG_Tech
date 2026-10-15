
import argparse
import datetime
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ─── Cấu hình ────────────────────────────────────────────────────────────────

BASE_URL = "https://cellphones.com.vn"
MAIN_LISTING = f"{BASE_URL}/mobile.html"
# Trang theo hãng - chỉ dùng thêm nếu trang chính chưa đủ số lượng
BRAND_SLUGS = ["apple", "samsung", "oppo", "xiaomi", "tecno", "honor", "nubia", "sony", "nokia",
               "infinix", "nothing-phone", "masstel", "realme", "itel", "huawei", "meizu", "vivo",
               "oneplus", "tcl", "benco", "asus"]

# Mục "Chọn theo nhu cầu" trên trang danh sách: mỗi trang liệt kê các máy cửa hàng xếp vào nhu cầu đó
NEED_PAGES = {
    "Chơi game":     f"{BASE_URL}/mobile/dien-thoai-gaming.html",
    "Pin trâu":      f"{BASE_URL}/mobile/dien-thoai-pin-trau.html",
    "5G":            f"{BASE_URL}/bo-loc/dien-thoai-5g",
    "Chụp ảnh đẹp":  f"{BASE_URL}/mobile/dien-thoai-chup-anh-quay-phim.html",
    "Điện thoại gập": f"{BASE_URL}/bo-loc/dien-thoai-man-hinh-gap-moi-nhat",
    "Điện thoại AI": f"{BASE_URL}/mobile/ai.html",
    "Phổ thông":     f"{BASE_URL}/mobile/dien-thoai-pho-thong.html",
}

CARD_SELECTOR = "a.product-info-container, .product-info a, h3.product-info-name a"
MIN_PHONE_PRICE = 500_000  # dưới mức này coi là giá rác (trả góp, đặt cọc...)

NON_PRODUCT_SLUGS = {"mobile", "laptop", "tablet", "dien-thoai", "may-tinh-bang"}
NON_PRODUCT_KEYWORDS = ["phu-kien", "thu-cu", "sforum", "tin-tuc", "khuyen-mai"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("crawl_phones")

# ─── Schema: 24 cột dữ liệu thô ──────────────────────────────────────────────
# Bộ cào chỉ lấy dữ liệu thô. Việc ghép thành đoạn văn để embedding (Row-to-Text)
# thuộc bước nạp dữ liệu của đồ án (mục 4.1.2), không làm ở đây.

BASE_COLUMNS = [
    "product_id",       # PK - slug URL
    "product_name",     # Tên sản phẩm sạch
    "brand",            # Hãng chuẩn hoá
    "url",              # URL đầy đủ
    "price",            # Giá bán (float)
    "ram_gb",           # RAM (float, GB)
    "storage_gb",       # Bộ nhớ trong (float, GB)
    "battery_mah",      # Dung lượng pin (float, mAh)
    "screen_size_inch", # Kích thước màn hình (float, inch)
    "rating",           # Đánh giá tổng hợp (float, 1-5)
    "key_features",     # Điểm nổi bật, cách nhau " | "
    "highlights",       # Đoạn mô tả dài
    "color_prices",     # JSON string {màu: giá}
    "specs_raw",        # JSON string toàn bộ bảng thông số
    "chipset",          # Chip, đúng chữ trên trang
    "screen_tech",      # Công nghệ màn hình
    "screen_resolution",# Độ phân giải màn hình
    "charging",         # Công nghệ sạc
    "camera_rear",      # Camera sau
    "camera_front",     # Camera trước
    "stock_status",     # Tình trạng hàng theo dữ liệu cấu trúc của trang: InStock, OutOfStock, PreOrder...
    "category_path",    # Đường dẫn danh mục trên trang, ví dụ "Điện thoại > Apple > iPhone 18 Series"
    "rating_count",     # Số lượt đánh giá đi kèm rating
    "needs",            # Nhu cầu mà cửa hàng gắn cho máy, ví dụ "Chơi game | Pin trâu"
]

# Các cột chữ lấy thẳng từ bảng thông số (tên cột = tên trường trong FIELD_RULES)
SPEC_TEXT_COLUMNS = ["chipset", "screen_tech", "screen_resolution", "charging",
                     "camera_rear", "camera_front"]

# Số tiền cụ thể: 30.490.000đ, 8.79 triệu, 1.5tr, 5.299 CNY... ("16 triệu màu" không phải giá)
MONEY_PATTERN = re.compile(
    r"\d{1,3}(?:[.,]\d{3}){1,3}\s*(?:đ|₫|vnđ|vnd|đồng)"
    r"|\d+(?:[.,]\d+)?\s*(?:tr|triệu)(?!\s*(?:màu|điểm|lượt|người))\b"
    r"|\d+(?:[.,]\d+)?\s*(?:usd|nhân dân tệ|tệ|yuan|cny)\b"
    r"|\$\s?\d",
    re.IGNORECASE,
)

# ─── Bảng nhãn -> trường nội bộ ──────────────────────────────────────────────

FIELD_RULES = [
    ("screen_size",       ["kích thước màn hình"]),
    ("screen_tech",       ["công nghệ màn hình"]),
    ("screen_resolution", ["độ phân giải"]),
    ("screen_features",   ["tính năng màn hình"]),
    ("refresh_rate",      ["tần số quét"]),
    ("camera_rear",       ["camera sau"]),
    ("camera_front",      ["camera trước"]),
    ("chipset",           ["chipset", "chip xử lý", "vi xử lý", "chip"]),
    ("ram",               ["dung lượng ram", "ram"]),
    ("storage",           ["bộ nhớ trong", "dung lượng lưu trữ", "rom"]),
    ("charging",          ["công nghệ sạc", "sạc"]),
    ("battery",           ["dung lượng pin", "pin"]),
]

BRAND_ALIAS = {"iphone": "Apple", "ipad": "Apple", "galaxy": "Samsung", "redmi": "Xiaomi",
               "poco": "Xiaomi", "rog phone": "ASUS", "zenfone": "ASUS", "pixel": "Google"}
BRAND_NAMES = ["Apple", "Samsung", "Xiaomi", "OPPO", "Tecno", "HONOR", "Nubia", "Sony", "Nokia",
               "Infinix", "Nothing Phone", "Masstel", "realme", "Itel", "Huawei", "Meizu", "vivo",
               "OnePlus", "TCL", "Benco", "ASUS", "Lenovo", "Motorola", "Google"]

# ─── Tiện ích làm sạch / chuẩn hoá ───────────────────────────────────────────

def clean(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def strip_markup(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return clean(text.replace("**", "").replace("\\n", " "))


def parse_price(text: str) -> Optional[float]:
    """RULE 1: Trả về float, bỏ mọi ký tự không phải số."""
    digits = re.sub(r"[^\d]", "", text or "")
    return float(digits) if digits else None


def to_gb(num: str, unit: str) -> float:
    """RULE 1: Trả về float GB."""
    n = float(num.replace(",", "."))
    return n * 1024.0 if unit.upper() == "TB" else n


def parse_gb(text: str) -> Optional[float]:
    """RULE 1 + RULE 4: float hoặc None, không bao giờ là chuỗi."""
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(TB|GB)", text or "", re.I)
    return to_gb(m.group(1), m.group(2)) if m else None


def parse_float(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text or "", re.I)
    return float(m.group(1).replace(",", ".")) if m else None


def parse_battery_mah(text: str) -> Optional[float]:
    """RULE 1: float mAh."""
    m = re.search(r"(\d[\d.,]*)\s*mAh", text or "", re.I)
    return float(re.sub(r"[^\d]", "", m.group(1))) if m else None


def infer_brand(name: str) -> str:
    low = (name or "").lower()
    for alias, brand in BRAND_ALIAS.items():
        if alias in low:
            return brand
    for b in BRAND_NAMES:
        if b.lower() in low:
            return b
    return name.split()[0] if name else "Unknown"


def map_label(label: str) -> Optional[str]:
    low = clean(label).lower()
    for field, keywords in FIELD_RULES:
        for kw in keywords:
            if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", low):
                return field
    return None


def has_money(text: str) -> bool:
    return bool(MONEY_PATTERN.search(text or ""))


def extract_spec_fields(specs_raw: dict) -> dict:
    """Ánh xạ bảng thông số thô -> các trường trong FIELD_RULES (mỗi trường lấy ô đầu tiên khớp).
    Ô có số tiền thì bỏ: có trang đặt bảng giá dưới tên giống thông số, ví dụ
    "Xiaomi 15 Pro RAM 12 GB + ROM 256 GB" -> "5.299 CNY (khoảng 18.7 triệu đồng)"."""
    fields = {}
    for label, value in specs_raw.items():
        field = map_label(label)
        if field and not fields.get(field) and not has_money(value):
            fields[field] = value
    return fields

# ─── Parse HTML (tách riêng để dễ test, không cần trình duyệt) ───────────────

def parse_json_ld(soup: BeautifulSoup) -> list:
    items = []
    for s in soup.find_all("script", type="application/ld+json"):
        raw = s.string or s.get_text()
        try:
            data = json.loads(raw, strict=False)
        except (json.JSONDecodeError, TypeError):
            continue
        stack = [data]
        while stack:
            d = stack.pop()
            if isinstance(d, list):
                stack.extend(d)
            elif isinstance(d, dict):
                items.append(d)
                if "@graph" in d:
                    stack.append(d["@graph"])
    return items


def ld_type_is(d: dict, name: str) -> bool:
    t = d.get("@type")
    return name in (t if isinstance(t, list) else [t])


def extract_stock_status(product_ld: Optional[dict]) -> Optional[str]:
    """Lấy offers.availability của JSON-LD Product, bỏ phần "https://schema.org/".
    Ví dụ máy đang bán: "InStock"; máy chỉ có trang thông tin: "OutOfStock"."""
    if not product_ld:
        return None
    offers = product_ld.get("offers")
    offers = offers[0] if isinstance(offers, list) and offers else offers
    if not isinstance(offers, dict) or not offers.get("availability"):
        return None
    return str(offers["availability"]).rstrip("/").split("/")[-1]


def extract_category_path(ld_items: list, product_name: str) -> Optional[str]:
    """Lấy JSON-LD BreadcrumbList, bỏ mục đầu (tên cửa hàng) và mục cuối (tên máy)."""
    for d in ld_items:
        if not ld_type_is(d, "BreadcrumbList"):
            continue
        names = []
        for e in d.get("itemListElement", []):
            item = e.get("item") if isinstance(e.get("item"), dict) else {}
            name = clean(e.get("name") or item.get("name") or "")
            if name:
                names.append(name)
        # Mục cuối là tên máy, có trang ghi thêm đuôi như "iPhone 17 Pro 256GB | Chính hãng"
        if names and names[-1].startswith(product_name):
            names = names[:-1]
        if names and names[0].upper() == "CELLPHONES":
            names = names[1:]
        return " > ".join(names) or None
    return None


def extract_specs(soup: BeautifulSoup) -> dict:
    """RULE 2+3: get_text(' ') strip HTML; trả về dict thô -> json.dumps() khi lưu."""
    specs = {}
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"], recursive=False)
        if len(cells) != 2:
            continue
        k = clean(cells[0].get_text(" "))
        vcell = cells[1]
        v = clean(vcell.get_text(" | ", strip=True)
                  if vcell.find(["br", "li", "p", "div"])
                  else vcell.get_text(" "))
        if k and v and len(k) < 60 and k.lower() not in ("thông số", "tiêu chí") and k not in specs:
            specs[k] = v
    return specs


def section_text(soup: BeautifulSoup, heading_kw: str, max_chars: int = 1500) -> str:
    """Gom văn bản nằm dưới 1 heading (h2/h3) cho tới heading h2 kế tiếp."""
    for h in soup.find_all(["h2", "h3"]):
        if heading_kw.lower() not in h.get_text().lower():
            continue
        node = h
        for _ in range(3):
            if node.find_next_sibling():
                break
            node = node.parent or node
        parts = []
        for sib in node.next_siblings:
            name = getattr(sib, "name", None)
            if not name:
                continue
            if name == "h2" or (name != "h2" and sib.find("h2")):
                break
            t = clean(sib.get_text(" "))
            if t:
                parts.append(t)
            if sum(map(len, parts)) > max_chars:
                break
        text = clean(" ".join(parts))
        if text:
            return text[:max_chars]
    return ""


def extract_price(soup: BeautifulSoup, product_ld: Optional[dict]) -> Optional[float]:
    """RULE 1: Trả về float giá hiện tại. RULE 4: None nếu không tìm thấy."""
    for sel in [".sale-price", "p.product-price", ".box-info__box-price .product__price--show",
                ".tpt---sale-price", ".product__price--show"]:
        el = soup.select_one(sel)
        p = parse_price(el.get_text()) if el else None
        if p and p >= MIN_PHONE_PRICE:
            return p
    m = re.search(r"(\d{1,3}(?:\.\d{3}){2,3})\s*(?:đ|₫|VND)", soup.get_text(" "))
    if m:
        p = parse_price(m.group(1))
        if p and p >= MIN_PHONE_PRICE:
            return p
    if product_ld:
        offers = product_ld.get("offers")
        offers = offers[0] if isinstance(offers, list) and offers else offers
        if isinstance(offers, dict):
            p = parse_price(str(offers.get("price") or offers.get("lowPrice") or ""))
            if p and p >= MIN_PHONE_PRICE:
                return p
    return None  # RULE 4


def extract_colors(soup: BeautifulSoup) -> dict:
    """Trả về dict {màu: giá float|None} -> json.dumps() ở ngoài (RULE 3)."""
    prices = {}
    for a in soup.find_all("a", href=re.compile(r"[?&]product_id=\d+")):
        name = clean(a.get("title") or "")
        if not name:
            strong = a.find(["strong", "b"])
            name = clean(strong.get_text(" ")) if strong else ""
        if not name or name in prices:
            continue
        mt = re.search(r"(\d{1,3}(?:\.\d{3}){1,3})\s*(?:₫|đ)", a.get_text(" "))
        prices[name] = parse_price(mt.group(1)) if mt else None
    return prices


def extract_key_features(soup: BeautifulSoup) -> list:
    """RULE 2: Trả về list str; caller sẽ join bằng ' | '."""
    for node in soup.find_all(string=re.compile(r"^\s*Tính năng nổi bật\s*$")):
        ul = node.find_next("ul")
        if not ul:
            continue
        items = [clean(li.get_text(" ")) for li in ul.find_all("li")]
        items = [i for i in items if i]
        if items and len(items) <= 8 and all(len(i) < 400 for i in items):
            return items
    return []


def parse_product_html(html: str, url: str) -> Optional[dict]:
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    full_name = clean(h1.get_text(" ")) if h1 else ""
    if not full_name:
        return None

    ld_items = parse_json_ld(soup)
    product_ld = next((d for d in ld_items if ld_type_is(d, "Product")), None)

    # Bảng thông số thô
    specs_raw = extract_specs(soup)

    # Ánh xạ nhãn -> trường (ô có số tiền bị bỏ)
    _internal = extract_spec_fields(specs_raw)

    # Tên & hãng
    name = clean(full_name.split("|")[0])
    brand = infer_brand(name)

    # Giá (RULE 1 -> float)
    price = extract_price(soup, product_ld)

    # Rating (RULE 1 -> float, RULE 4 -> None)
    rating: Optional[float] = None
    if product_ld and isinstance(product_ld.get("aggregateRating"), dict):
        ar = product_ld["aggregateRating"]
        rv = ar.get("ratingValue")
        try:
            rating = float(rv) if rv is not None else None
        except (ValueError, TypeError):
            rating = None
    if rating is None:
        body_text = soup.get_text(" ")
        m_r = re.search(r"(\d(?:[.,]\d)?)\s*\((\d+)\s*đánh giá\)", body_text)
        if m_r:
            try:
                rating = float(m_r.group(1).replace(",", "."))
            except (ValueError, TypeError):
                rating = None

    # Số GB (RULE 1 -> float)
    tokens = re.findall(r"(\d+(?:[.,]\d+)?)\s*(GB|TB)", name, re.I)
    ram_gb     = parse_gb(_internal.get("ram", ""))
    storage_gb = parse_gb(_internal.get("storage", ""))
    if not storage_gb and tokens:
        storage_gb = to_gb(*tokens[-1])
    if not ram_gb and len(tokens) >= 2:
        ram_gb = to_gb(*tokens[0])

    # Pin & màn hình (RULE 1 -> float)
    battery_mah      = parse_battery_mah(_internal.get("battery", ""))
    screen_size_inch  = parse_float(_internal.get("screen_size", ""), r"(\d+(?:[.,]\d+)?)")

    # Màu & giá (RULE 3)
    color_prices_dict = extract_colors(soup)
    color_prices_json: Optional[str] = (
        json.dumps(color_prices_dict, ensure_ascii=False) if color_prices_dict else None
    )

    # Tính năng nổi bật (RULE 2)
    raw_features = extract_key_features(soup)
    key_features: Optional[str] = " | ".join(raw_features) if raw_features else None

    # Highlights (RULE 2: bỏ HTML + RULE 4: không để chuỗi rỗng)
    highlights_text = re.sub(r"https?://\S+", "", section_text(soup, "Đặc điểm nổi bật")).strip()
    if not highlights_text:
        md = soup.find("meta", attrs={"name": "description"})
        highlights_text = clean(md.get("content")) if md else ""
    if key_features and highlights_text:
        h = highlights_text
        for f in key_features.split(" | "):
            h = h.replace(f, "")
        highlights_text = re.sub(r"^\s*Tính năng nổi bật\s*", "", clean(h))
    highlights: Optional[str] = highlights_text or None  # RULE 4

    # specs_raw JSON (RULE 3)
    specs_raw_json: Optional[str] = (
        json.dumps(specs_raw, ensure_ascii=False) if specs_raw else None
    )

    # Tổng hợp record 20 cột
    rec: dict = {
        "product_id":       Path(urlparse(url).path).stem,
        "product_name":     name,
        "brand":            brand,
        "url":              url,
        "price":            price,             # float | None  (RULE 1, 4)
        "ram_gb":           ram_gb,            # float | None
        "storage_gb":       storage_gb,        # float | None
        "battery_mah":      battery_mah,       # float | None
        "screen_size_inch": screen_size_inch,  # float | None
        "rating":           rating,            # float | None
        "key_features":     key_features,      # str " | " | None  (RULE 2)
        "highlights":       highlights,        # str | None  (RULE 4)
        "color_prices":     color_prices_json, # JSON str | None  (RULE 3)
        "specs_raw":        specs_raw_json,    # JSON str | None  (RULE 3)
    }

    # 6 cột thông số dạng chữ, giữ đúng chữ trên trang; không có thì None (RULE 4)
    for col in SPEC_TEXT_COLUMNS:
        rec[col] = _internal.get(col) or None

    # Tình trạng hàng, danh mục, số lượt đánh giá lấy từ JSON-LD của trang
    rec["stock_status"] = extract_stock_status(product_ld)
    rec["category_path"] = extract_category_path(ld_items, name)
    rating_count = None
    if product_ld and isinstance(product_ld.get("aggregateRating"), dict):
        try:
            rating_count = int(product_ld["aggregateRating"].get("reviewCount"))
        except (ValueError, TypeError):
            rating_count = None
    rec["rating_count"] = rating_count
    rec["needs"] = None  # gắn sau, khi đã gom URL của các trang nhu cầu (collect_needs)

    # Trường kiểm tra nội bộ (gating chất lượng) - sẽ bị lọc bởi export_all()
    rec["_chipset"]   = _internal.get("chipset")
    rec["_ram"]       = _internal.get("ram")
    rec["_storage"]   = _internal.get("storage")
    rec["_specs_len"] = len(specs_raw)

    return rec

# ─── Selenium ────────────────────────────────────────────────────────────────

def create_driver(headless: bool) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.page_load_strategy = "eager"
    try:
        driver = webdriver.Chrome(options=opts)
    except WebDriverException:
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.set_page_load_timeout(30)
    return driver


def is_product_url(href: str) -> bool:
    if not href or "?" in href or "#" in href:
        return False
    p = urlparse(href)
    if p.netloc and "cellphones.com.vn" not in p.netloc:
        return False
    segs = [s for s in p.path.split("/") if s]
    if len(segs) != 1 or not segs[0].endswith(".html"):
        return False
    slug = segs[0][:-5]
    return slug not in NON_PRODUCT_SLUGS and not any(k in slug for k in NON_PRODUCT_KEYWORDS)


def collect_urls(driver, listing_url: str, limit: Optional[int], known: list) -> list:
    """Gom URL sản phẩm. limit=None nghĩa là không giới hạn (lấy hết trang)."""
    limit_str = str(limit) if limit is not None else "tất cả"
    log.info(f"Gom URL từ {listing_url} (đã có {len(known)}/{limit_str})")
    driver.get(listing_url)
    time.sleep(4)
    found, stale = list(known), 0
    while stale < 5:
        if limit is not None and len(found) >= limit:
            break
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 1500);")
        time.sleep(1.5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
        before = len(found)
        for a in driver.find_elements(By.CSS_SELECTOR, CARD_SELECTOR):
            href = a.get_attribute("href")
            if is_product_url(href) and href not in found:
                found.append(href)
        log.info(f"  -> {len(found)}/{limit_str}")
        # stale chỉ reset khi TÌM ĐƯỢC link mới, không phụ thuộc vào việc click nút
        if len(found) > before:
            stale = 0
        else:
            stale += 1
        try:
            btn = driver.find_element(By.CSS_SELECTOR, ".btn-show-more")
            # Chỉ dùng JavaScript click - tránh lỗi is_displayed() sai trong headless mode
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
            time.sleep(0.8)
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(2.5)
        except Exception:
            # Nút không còn trong DOM -> thực sự hết trang
            log.info("  Không tìm thấy nút 'Xem thêm', đã hết sản phẩm.")
            break
    return found if limit is None else found[:limit]


def scroll_page(driver):
    for frac in (0.25, 0.5, 0.75, 1.0):
        driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight * {frac});")
        time.sleep(0.6)


def click_spec_button(driver) -> bool:
    return bool(driver.execute_script("""
        const els = [...document.querySelectorAll('button,a,div')].filter(e => {
            const t = (e.innerText || '').trim();
            return t.length < 40 && /cấu hình chi tiết|xem thêm cấu hình|thông số kỹ thuật/i.test(t);
        });
        if (els.length) { els[0].click(); return true; } return false;
    """))


def scrape_phone(driver, url: str, delay: float, debug_dir: Optional[Path]) -> Optional[dict]:
    for attempt in (1, 2):
        try:
            driver.get(url)
            try:
                WebDriverWait(driver, 12).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))
            except TimeoutException:
                pass
            scroll_page(driver)
            rec = parse_product_html(driver.page_source, url)
            if rec and rec["_specs_len"] < 6:
                if click_spec_button(driver):
                    time.sleep(1.5)
                    rec = parse_product_html(driver.page_source, url) or rec
            if rec and rec["_specs_len"] < 3 and debug_dir:
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / f"{rec['product_id']}.html").write_text(driver.page_source, encoding="utf-8")
            time.sleep(delay)
            return rec
        except (TimeoutException, WebDriverException) as e:
            log.warning(f"  Lần {attempt} lỗi {url}: {str(e)[:80]}")
            time.sleep(3)
    return None

# ─── Lưu / đọc tiến độ / xuất file ───────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rec: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def collect_needs(driver, out: Path) -> dict:
    """Gom URL sản phẩm của từng trang "Chọn theo nhu cầu" -> {nhãn: [url]}.
    Lưu ra need_urls.json; lần chạy sau đọc lại file này thay vì gom lại."""
    need_file = out / "need_urls.json"
    if need_file.exists():
        return json.loads(need_file.read_text(encoding="utf-8"))
    need_urls = {}
    for label, page in NEED_PAGES.items():
        need_urls[label] = collect_urls(driver, page, None, [])
        log.info(f"  Nhu cầu '{label}': {len(need_urls[label])} máy")
    need_file.write_text(json.dumps(need_urls, ensure_ascii=False, indent=1), encoding="utf-8")
    return need_urls


def attach_needs(records: list, need_urls: dict) -> None:
    """Gắn cột needs cho từng bản ghi theo thứ tự nhãn trong NEED_PAGES, ví dụ "Chơi game | 5G"."""
    for rec in records:
        labels = [label for label in NEED_PAGES if rec["url"] in need_urls.get(label, [])]
        rec["needs"] = " | ".join(labels) if labels else None


def export_all(records: list, out: Path) -> pd.DataFrame:
    """
    Xuất đúng các cột trong BASE_COLUMNS.
    phones.csv  : UTF-8 BOM
    phones.xlsx : cần openpyxl
    phones.json : cùng các cột (không kèm các trường kiểm tra bắt đầu bằng "_")
    """
    rows = [{col: r.get(col) for col in BASE_COLUMNS} for r in records]
    df = pd.DataFrame(rows, columns=BASE_COLUMNS)

    df.to_csv(out / "phones.csv", index=False, encoding="utf-8-sig")
    try:
        df.to_excel(out / "phones.xlsx", index=False)
    except ImportError:
        log.warning("Chưa cài openpyxl nên bỏ qua phones.xlsx (pip install openpyxl)")

    (out / "phones.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"Đã xuất {len(df)} điện thoại -> {out}/ ({len(BASE_COLUMNS)} cột)")
    return df


def quality_report(df: pd.DataFrame):
    """Báo cáo tỉ lệ đầy đủ dữ liệu cho các cột trong BASE_COLUMNS."""
    print("\n=== TỈ LỆ DỮ LIỆU ĐẦY ĐỦ THEO CỘT ===")
    skip = {"url", "product_id", "product_name", "brand"}
    for col in BASE_COLUMNS:
        if col in skip:
            continue
        pct = df[col].notna().mean() * 100 if col in df else 0.0
        print(f"  {col:<20} {pct:5.1f}%  {'#' * int(pct // 5)}")

    key_numeric = ["price", "ram_gb", "storage_gb", "battery_mah", "screen_size_inch"]
    bad = df[df[key_numeric].isna().any(axis=1)]
    if len(bad):
        print(f"\n{len(bad)} máy thiếu ít nhất 1 trường số chính:")
        for _, r in bad.head(10).iterrows():
            missing = [k for k in key_numeric if pd.isna(r[k])]
            print(f"  - {r['product_name']}: thiếu {missing}")

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=300,
                    help="số điện thoại tối đa (bị ghi đè bởi --all)")
    ap.add_argument("--all", action="store_true",
                    help="lấy TẤT CẢ sản phẩm trên cửa hàng (không giới hạn)")
    ap.add_argument("--test", type=int, default=0, help="chỉ cào N máy để thử")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--out", default="data_phones")
    ap.add_argument("--no-brand-pages", action="store_true",
                    help="không dùng thêm trang theo hãng")
    ap.add_argument("--no-needs", action="store_true",
                    help="không gom các trang 'Chọn theo nhu cầu' (cột needs để trống)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl, urls_file = out / "phones.jsonl", out / "phone_urls.json"

    # limit=None nghĩa là không giới hạn
    if args.test:
        limit: Optional[int] = args.test
    elif args.all:
        limit = None
    else:
        limit = args.max

    driver = create_driver(args.headless)
    try:
        urls = json.loads(urls_file.read_text(encoding="utf-8")) if urls_file.exists() else []
        need_more = (limit is None) or (len(urls) < limit)
        if need_more:
            urls = collect_urls(driver, MAIN_LISTING, limit, urls)
            # Chỉ cần gom thêm trang hãng nếu CHƯA ĐỦ SỐ LƯỢNG (limit có giá trị)
            # Khi limit=None (--all), trang chính đã có tất cả sản phẩm, không cần thêm
            still_need = (limit is not None) and (len(urls) < limit)
            if still_need and not args.no_brand_pages:
                for b in BRAND_SLUGS:
                    urls = collect_urls(driver, f"{BASE_URL}/mobile/{b}.html", limit, urls)
                    if limit is not None and len(urls) >= limit:
                        break
            urls_file.write_text(json.dumps(urls, ensure_ascii=False, indent=1), encoding="utf-8")
        if limit is not None:
            urls = urls[:limit]
        log.info(f"Tổng {len(urls)} URL điện thoại")

        records = load_jsonl(jsonl)
        done = {r["url"] for r in records}
        todo = [u for u in urls if u not in done]
        log.info(f"Đã có {len(done)} máy, còn {len(todo)} máy cần cào")

        for url in tqdm(todo, desc="Điện thoại"):
            rec = scrape_phone(driver, url, args.delay, out / "debug_html")
            if rec is None:
                log.warning(f"  Bỏ qua (parse thất bại hoàn toàn): {url}")
                continue
            append_jsonl(jsonl, rec)
            records.append(rec)
            log.info(
                f"  ok {rec['product_name'][:45]} | "
                f"{rec['price']} d | "
                f"{rec['_specs_len']} thong so"
            )

        need_urls = {} if args.no_needs else collect_needs(driver, out)
    finally:
        driver.quit()

    records = load_jsonl(jsonl)
    if records:
        attach_needs(records, need_urls)
        quality_report(export_all(records, out))


if __name__ == "__main__":
    main()
