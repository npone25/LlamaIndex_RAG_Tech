# Mục 3.2.2 - Hello World: Document -> VectorStoreIndex (trong RAM) -> Query Engine
# Chạy: python hello_world.py   (cần GOOGLE_API_KEY trong .env; không cần PostgreSQL)
# Dùng đúng mô hình nhúng và LLM của đồ án, dữ liệu là một dòng thật trong phones.xlsx

import _paths  # noqa: F401  (thêm thư mục gốc vào đường dẫn tìm module)
import pandas as pd
from llama_index.core import VectorStoreIndex

from phone_advisor.ingestion.documents import PHONE_FILE, row_to_document
from phone_advisor.llm import create_answer_llm
from phone_advisor.ingestion.embeddings import create_embed_model

df = pd.read_excel(PHONE_FILE)
fetched = df[df["product_name"] == "Samsung Galaxy Z Fold7 12GB 256GB"].iloc[0]
doc = row_to_document(fetched)
print("Document:", doc.get_content()[:100], "...")

index = VectorStoreIndex.from_documents([doc], embed_model=create_embed_model())
query_engine = index.as_query_engine(llm=create_answer_llm())
response = query_engine.query("Galaxy Z Fold7 dùng chip gì?")
print("Trả lời:", str(response).strip())
print("Nguồn :", response.source_nodes[0].node.metadata["product_name"],
      "| điểm %.3f" % response.source_nodes[0].score)
