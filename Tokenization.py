import tiktoken

# Load a model multilingual BPE tokenizer
enc = tiktoken.get_encoding("o200k_base")


# Tokenize a string
def inspect_tokenization(text):
    # Stage 1: RAW UTF-8 bytes
    raw_bytes = text.encode("utf-8")

    # Stage 2&3: BPE merge + Vocabulary ID lookup
    token_ids = enc.encode(text)
    token_pieces = [enc.decode_single_token_bytes(tid) for tid in token_ids]

    print(f"Input text: {text}")
    print(f"Stage 1 (UTF-8 Hex): {' '.join(f'{b:02x}' for b in raw_bytes)}")
    print(f"Stage 2 (Subwords): {[p.decode('utf-8') for p in token_pieces]}")
    print(f"Stage 3 (Token IDs): {token_ids}")


inspect_tokenization("I love tokenization!")
inspect_tokenization("我喜欢猫。")
