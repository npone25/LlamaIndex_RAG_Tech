"""Script kiểm chứng tính tương đương giữa luồng gốc và chế độ truy vết (mục 2, ràng buộc 3).

Chạy trên dữ liệu thật với ít nhất 5 câu hỏi: thông số, chính sách, ngoài phạm vi.
So sánh danh sách node_id sau reranker và ngưỡng giữa:
- Luồng gốc: tao_query_engine(...) hiện tại
- Chế độ truy vết: truy_vet_cau_hoi(...)
In ra từng câu: TRÙNG hoặc KHÁC.
"""

import _duong_dan  # noqa: F401  (thêm thư mục gốc và loi_rag/ vào đường dẫn tìm module)
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from he_thong.cau_hinh import lay_nguong_tu_choi
from he_thong.truy_vet import khoi_tao_tai_nguyen_truy_vet, truy_vet_cau_hoi
from he_thong.dich_vu import xay_dung_tro_ly_thuc
from tro_ly import tao_query_engine


DANH_SACH_CAU_HOI = [
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
    tro_ly = xay_dung_tro_ly_thuc()
    tai_nguyen = khoi_tao_tai_nguyen_truy_vet(tro_ly=tro_ly)
    query_engine_goc = tro_ly.query_engine
    if query_engine_goc is None:
        query_engine_goc = tao_query_engine(
            llm=tro_ly.llm,
            embed_model=tro_ly.embed_model,
            retriever=tro_ly.retriever,
            hau_xu_ly=tro_ly.hau_xu_ly,
            nodes_theo_bang=tro_ly.nodes_theo_bang,
        )
    print(f"   Khởi tạo xong trong {time.perf_counter() - t0:.2f}s\n")

    tat_ca_trung = True
    so_cau_so = 0
    so_cau_trung = 0

    for i, cau_hoi in enumerate(DANH_SACH_CAU_HOI, 1):
        print(f"--- Câu {i}: \"{cau_hoi}\" ---")

        # 1. Chạy luồng gốc (tao_query_engine)
        # Lưu ý: query_engine.query sẽ chạy retrieval + rerank + postprocessor + synthesize
        t_goc_0 = time.perf_counter()
        resp_goc = query_engine_goc.query(cau_hoi)
        t_goc_ms = (time.perf_counter() - t_goc_0) * 1000

        nodes_goc = getattr(resp_goc, "source_nodes", []) or []
        ids_goc = [nws.node.node_id for nws in nodes_goc]

        # 2. Chạy chế độ truy vết (truy_vet_cau_hoi)
        t_tv_0 = time.perf_counter()
        # Để so sánh công bằng với query_engine_rag, ta chạy qua truy_vet_cau_hoi
        vet = truy_vet_cau_hoi(cau_hoi, tai_nguyen=tai_nguyen, bat_loi=False, luu_file=False)
        t_tv_ms = (time.perf_counter() - t_tv_0) * 1000

        # Nếu câu bị guardrails chặn ngay (như câu ngoài phạm vi), số node nguồn là rỗng []
        # Tương ứng với việc query_engine gốc cũng bị loại hết hoặc từ chối
        ids_truy_vet = [n["node_id"] for n in vet["nguon"]]

        dinh_tuyen = vet.get("dinh_tuyen") or {}
        cac_cong_cu = dinh_tuyen.get("cac_cong_cu") or [dinh_tuyen.get("cong_cu")]
        bi_chan = (vet.get("bao_ve") or {}).get("cho_phep") is False
        print(f"   • Công cụ router chọn: {cac_cong_cu}")
        if bi_chan:
            # Truy vết đi qua lớp bảo vệ như app thật; luồng RAG gốc trong script này không có lớp bảo vệ
            print(f"   • Luồng gốc RAG, không có lớp bảo vệ ({round(t_goc_ms)} ms) : {ids_goc}")
            print(f"   • Truy vết: bị lớp bảo vệ chặn ({(vet.get('bao_ve') or {}).get('lop')}), không truy hồi")
            print("   => ĐÁNH GIÁ: KHÔNG SO (câu bị chặn trước khâu truy hồi)\n")
            continue
        if "tu_van_va_chinh_sach" not in cac_cong_cu:
            # Router không chọn nhánh RAG nên truy vết không có node để so với luồng RAG gốc
            print(f"   • Luồng gốc ({round(t_goc_ms)} ms) : {ids_goc}")
            print("   => ĐÁNH GIÁ: KHÔNG SO (router chọn công cụ khác RAG)\n")
            continue

        # Kiểm tra kết quả
        trung_khop = (ids_goc == ids_truy_vet)
        ket_qua_str = "TRÙNG" if trung_khop else "KHÁC"

        if not trung_khop:
            # Trường hợp câu ngoài phạm vi: nếu guardrails chặn thì ids_truy_vet=[]
            # Kiểm tra xem query_engine gốc có node nào vượt ngưỡng không
            if not ids_truy_vet and not ids_goc:
                trung_khop = True
                ket_qua_str = "TRÙNG (Cùng từ chối / 0 node)"

        so_cau_so = so_cau_so + 1
        if trung_khop:
            so_cau_trung = so_cau_trung + 1
        if not trung_khop:
            tat_ca_trung = False

        print(f"   • Luồng gốc ({round(t_goc_ms)} ms) : {ids_goc}")
        print(f"   • Truy vết  ({round(t_tv_ms)} ms) : {ids_truy_vet}")
        print(f"   => ĐÁNH GIÁ: {ket_qua_str}\n")

    print("=" * 70)
    print(f"Số câu được so (đi qua nhánh RAG): {so_cau_so} | Trùng: {so_cau_trung}/{so_cau_so}")
    if tat_ca_trung:
        print("KẾT LUẬN: TẤT CẢ CÁC CÂU ĐỀU TRÙNG KHỚP HOÀN TOÀN! (ĐẠT CHUẨN MỤC 2, RÀNG BUỘC 3)")
    else:
        print("KẾT LUẬN: CÓ CÂU KHÔNG TRÙNG KHỚP! VUI LÒNG KIỂM TRA LẠI.")
    print("=" * 70)

    if not tat_ca_trung:
        sys.exit(1)


if __name__ == "__main__":
    main()
