from llama_index.core.utils import get_tokenizer

counts = get_tokenizer()
for sentence in ["Battery lasts all day", "Pin dùng được cả ngày"]:
    print(sentence, "->", len(counts(sentence)), "token")