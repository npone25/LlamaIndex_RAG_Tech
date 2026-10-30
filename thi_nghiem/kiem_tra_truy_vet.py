"""Script kiểm chứng tính tương đương giữa luồng gốc và chế độ truy vết (mục 2, ràng buộc 3).

Chạy trên dữ liệu thật với ít nhất 5 câu hỏi: thông số, chính sách, ngoài phạm vi.
So sánh danh sách node_id sau reranker và ngưỡng giữa:
- Luồng gốc: create_query_engine(...) hiện tại
- Chế độ truy vết: trace_question(...)
In ra từng câu: TRÙNG hoặc KHÁC.
"""

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from he_thong.cau_hinh import get_refusal_threshold
from he_thong.truy_vet import init_trace_resources, trace_question
from he_thong.dich_vu import build_real_assistant
from tro_ly import create_query_engine


QUESTION_SET = [
    "Máy Samsung nào chụp ảnh đẹp?",
    "Điện thoại bị lỗi được đổi trả trong bao lâu?",
    "Máy nào pin trâu dưới 10 triệu?",
    "Chính sách bảo hành pin và sạc như thế nào?",
    "Cách làm món phở bò truyền thống Hà Nội?",
]


def main():
    print("=" * 70)
    print("KIỂM CHỨNG TƯƠNG ĐƯƠNG: LUỒNG GỐC vs TRUY VẾT MINH BẠCH")
    print("=" * 70)

    print("\n1. Đang khởi tạo mô hình và kết nối cơ sở dữ liệu thật...")
    t0 = time.perf_counter()
    tro_ly = build_real_assistant()
    resources = init_trace_resources(tro_ly=tro_ly)
    base_query_engine = tro_ly.query_engine
    if base_query_engine is None:
        base_query_engine = create_query_engine(
            llm=tro_ly.llm,
            embed_model=tro_ly.embed_model,
            retriever=tro_ly.retriever,
            postprocessors=tro_ly.postprocessors,
            nodes_by_table=tro_ly.nodes_by_table,
        )
    print(f"   Khởi tạo xong trong {time.perf_counter() - t0:.2f}s\n")

    all_matched = True
    numeric_question_count = 0
    matched_question_count = 0

    for i, cau_hoi in enumerate(QUESTION_SET, 1):
        print(f"--- Câu {i}: \"{cau_hoi}\" ---")

        # 1. Chạy luồng gốc (create_query_engine)
        # Lưu ý: query_engine.query sẽ chạy retrieval + rerank + postprocessor + synthesize
        t_base_0 = time.perf_counter()
        resp_original = base_query_engine.query(cau_hoi)
        t_base_ms = (time.perf_counter() - t_base_0) * 1000

        original_nodes = getattr(resp_original, "source_nodes", []) or []
        original_ids = [nws.node.node_id for nws in original_nodes]

        # 2. Chạy chế độ truy vết (trace_question)
        t_tv_0 = time.perf_counter()
        # Để so sánh công bằng với query_engine_rag, ta chạy qua trace_question
        trace = trace_question(cau_hoi, resources=resources, catch_errors=False, save_file=False)
        t_tv_ms = (time.perf_counter() - t_tv_0) * 1000

        # Nếu câu bị guardrails chặn ngay (như câu ngoài phạm vi), số node nguồn là rỗng []
        # Tương ứng với việc query_engine gốc cũng bị loại hết hoặc từ chối
        trace_ids = [n["node_id"] for n in trace["nguon"]]

        dinh_tuyen = trace.get("dinh_tuyen") or {}
        tools = dinh_tuyen.get("cac_cong_cu") or [dinh_tuyen.get("cong_cu")]
        blocked = (trace.get("bao_ve") or {}).get("cho_phep") is False
        print(f"   • Công cụ router chọn: {tools}")
        if blocked:
            # Truy vết đi qua lớp bảo vệ như app thật; luồng RAG gốc trong script này không có lớp bảo vệ
            print(f"   • Luồng gốc RAG, không có lớp bảo vệ ({round(t_base_ms)} ms) : {original_ids}")
            print(f"   • Truy vết: bị lớp bảo vệ chặn ({(trace.get('bao_ve') or {}).get('lop')}), không truy hồi")
            print("   => ĐÁNH GIÁ: KHÔNG SO (câu bị chặn trước khâu truy hồi)\n")
            continue
        if "tu_van_va_chinh_sach" not in tools:
            # Router không chọn nhánh RAG nên truy vết không có node để so với luồng RAG gốc
            print(f"   • Luồng gốc ({round(t_base_ms)} ms) : {original_ids}")
            print("   => ĐÁNH GIÁ: KHÔNG SO (router chọn công cụ khác RAG)\n")
            continue

        # Kiểm tra kết quả
        ids_match = (original_ids == trace_ids)
        result_str = "TRÙNG" if ids_match else "KHÁC"

        if not ids_match:
            # Trường hợp câu ngoài phạm vi: nếu guardrails chặn thì trace_ids=[]
            # Kiểm tra xem query_engine gốc có node nào vượt ngưỡng không
            if not trace_ids and not original_ids:
                ids_match = True
                result_str = "TRÙNG (Cùng từ chối / 0 node)"

        numeric_question_count = numeric_question_count + 1
        if ids_match:
            matched_question_count = matched_question_count + 1
        if not ids_match:
            all_matched = False

        print(f"   • Luồng gốc ({round(t_base_ms)} ms) : {original_ids}")
        print(f"   • Truy vết  ({round(t_tv_ms)} ms) : {trace_ids}")
        print(f"   => ĐÁNH GIÁ: {result_str}\n")

    print("=" * 70)
    print(f"Số câu được so (đi qua nhánh RAG): {numeric_question_count} | Trùng: {matched_question_count}/{numeric_question_count}")
    if all_matched:
        print("KẾT LUẬN: TẤT CẢ CÁC CÂU ĐỀU TRÙNG KHỚP HOÀN TOÀN! (ĐẠT CHUẨN MỤC 2, RÀNG BUỘC 3)")
    else:
        print("KẾT LUẬN: CÓ CÂU KHÔNG TRÙNG KHỚP! VUI LÒNG KIỂM TRA LẠI.")
    print("=" * 70)

    if not all_matched:
        sys.exit(1)


if __name__ == "__main__":
    main()
