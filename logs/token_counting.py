from llama_index.core.utils import get_tokenizer

dem = get_tokenizer()
for cau in ["Battery lasts all day", "Pin dùng được cả ngày"]:
    print(cau, "->", len(dem(cau)), "token")