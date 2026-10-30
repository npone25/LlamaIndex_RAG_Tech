# Thí nghiệm 8.2.1 - NLSQLTableQueryEngine: câu hỏi tự nhiên -> SQL trên bảng products
# Chạy: python exp_821_nlsql.py   (cần seed_sql.py đã chạy và GOOGLE_API_KEY)

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
from sqlalchemy import text

from dinh_tuyen import tao_engine_chi_doc, tao_sql_engine
from llm_gemini import create_answer_llm

sql_engine = tao_sql_engine(create_answer_llm())
# Mỗi câu hỏi kèm câu SQL viết tay để đối chiếu kết quả
KIEM_TRA = [
    ("3 điện thoại Samsung còn hàng rẻ nhất là máy nào, giá bao nhiêu?",
     "SELECT product_name, price FROM products WHERE brand = 'Samsung' AND stock_status = "
     "'InStock' AND price IS NOT NULL ORDER BY price LIMIT 3"),
    ("Giá trung bình của các điện thoại Xiaomi có giá là bao nhiêu?",
     "SELECT round(avg(price)) FROM products WHERE brand = 'Xiaomi' AND price IS NOT NULL"),
    ("Có bao nhiêu điện thoại pin từ 6000 mAh trở lên?",
     "SELECT count(*) FROM products WHERE battery_mah >= 6000"),
]

for i, (cau_hoi, sql_tay) in enumerate(KIEM_TRA, start=1):
    print("=== CÂU %d: %s ===" % (i, cau_hoi))
    response = sql_engine.query(cau_hoi)
    print("SQL do LLM sinh:", " ".join(response.metadata["sql_query"].split()))
    print("Kết quả SQL    :", str(response.metadata.get("result"))[:120])
    with tao_engine_chi_doc().connect() as conn:
        print("SQL viết tay   :", [tuple(d) for d in conn.execute(text(sql_tay)).fetchall()])
    print("Trả lời        :", str(response).replace("\n", " ")[:120])

print("=== CA LỖI: HỎI THÔNG TIN KHÔNG CÓ TRONG BẢNG (SỐ LƯỢNG TỒN KHO) ===")
response = sql_engine.query("Cửa hàng còn bao nhiêu chiếc Galaxy S25 Ultra trong kho?")
print("SQL do LLM sinh:", " ".join(response.metadata["sql_query"].split()))
print("Trả lời        :", str(response).replace("\n", " ")[:120])
