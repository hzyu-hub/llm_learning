import torch
from torch import nn

torch.manual_seed(42)

# Tiny toy vocabulary & model dimensions
vocab = {"dog": 0, "bites": 1, "man": 2}
vocab_size = len(vocab)
d_model = 4
max_seq_len = 10

# Stage 4: Token Embedding lookup table(vocab_size x d_model)
token_embedding = nn.Embedding(vocab_size, d_model)
positional_embedding = nn.Embedding(max_seq_len, d_model)


def encode_inputs(words):

    # Convert words to token IDs
    token_ids = torch.tensor([vocab[word] for word in words])

    positions = torch.arange(len(words))

    # Lookup token embeddings
    token_vecs = token_embedding(token_ids)

    # Lookup positional embeddings
    pos_vecs = positional_embedding(positions)

    return token_vecs + pos_vecs


sent_1 = encode_inputs(["dog", "bites", "man"])
sent_2 = encode_inputs(["man", "bites", "dog"])

print("Vector for 'dog' at pos = 0 (subject):", sent_1[0].detach().numpy().round(3))
print("Vector for 'dog' at pos = 2 (object):", sent_2[2].detach().numpy().round(3))
print("Are they identical?", torch.allclose(sent_1[0], sent_2[2]))
