import torch
from Multi_Head_Attention import multi_head_attention
from Residual_Connection_LayerNorm import residual_connection_and_layernorm
from torch import nn


def feed_forward_network(x, w_1, b_1, w_2, b_2):
    """
    Implements a feed forward network with two linear layers and a ReLU activation function.

    Args:
        x: Input tensor of shape (batch_size, seq_length, d_model)
        w_1: Weight tensor for the first linear layer of shape (d_model, d_ff)
        b_1: Bias tensor for the first linear layer of shape (d_ff,)
        w_2: Weight tensor for the second linear layer of shape (d_ff, d_model)
        b_2: Bias tensor for the second linear layer of shape (d_model,)

    Returns:
        Output tensor of shape (batch_size, seq_length, d_model)
    """
    # Expand each token from 4 features to 16 fearues
    linear_1 = x @ w_1 + b_1  # Linear transformation
    relu = torch.relu(linear_1)  # ReLU activation
    linear_2 = relu @ w_2 + b_2  # Second linear transformation

    return linear_2  # Output of the feed forward network


if __name__ == "__main__":
    # Example usage
    torch.manual_seed(0)

    d_model = 4
    num_heads = 2

    # 1. Tokenize the sentence using a simple world-level tokenizer
    sentence = "I love China"
    tokens = sentence.split()  # ['I', 'love', 'China']

    # 2. Convert the tokens into integer vocabulary IDs.
    vocab = {"I": 0, "love": 1, "China": 2}
    tokens_ids = torch.tensor([vocab[token] for token in tokens])  # [0, 1, 2]
    seq_length = tokens_ids.shape[-1]  # 3

    # 3. Convert the integer IDs into embeddings.
    embedding_dim = d_model
    token_embedding = nn.Embedding(len(vocab), embedding_dim)
    word_vectors = token_embedding(tokens_ids).squeeze(0)  # (1, seq_length, d_model)

    # 4.Look up a trainable position vector for each token ID
    position_embedding = nn.Embedding(seq_length, embedding_dim)
    position_ids = torch.arange(seq_length).unsqueeze(0)  # (1, seq_length)
    position_vectors = position_embedding(position_ids)

    # 5. Add token and position vectors to construct the attetion input
    x = word_vectors + position_vectors  # (batch_size, seq_length, d_model)

    # 6. Initialize trainable attention projection matrices
    w_q = nn.Parameter(torch.randn(d_model, d_model))
    w_k = nn.Parameter(torch.randn(d_model, d_model))
    w_v = nn.Parameter(torch.randn(d_model, d_model))
    w_o = nn.Parameter(torch.randn(d_model, d_model))

    # 7. Compute multi-head attention
    attention_output, attn_weights = multi_head_attention(
        x, w_q, w_k, w_v, w_o, num_heads
    )

    # 8. Apply residual connection and layer normalization
    gamma = nn.Parameter(torch.ones(d_model))  # Scale parameter
    beta = nn.Parameter(torch.zeros(d_model))  # Bias parameter
    eps = 1e-5
    residual_layernorm_output = residual_connection_and_layernorm(
        x, attention_output, gamma, beta, eps
    )

    # 9. Apply feed forward network
    d_ff = 16  # Dimension of the feed forward network
    w_1 = nn.Parameter(torch.randn(d_model, d_ff))  # Weight for the first linear layer
    b_1 = nn.Parameter(torch.randn(d_ff))  # Bias for the first linear layer
    w_2 = nn.Parameter(torch.randn(d_ff, d_model))  # Weight for the second linear layer
    b_2 = nn.Parameter(torch.randn(d_model))  # Bias for the second linear layer
    feed_forward_output = feed_forward_network(
        residual_layernorm_output, w_1, b_1, w_2, b_2
    )

    print("feed_forward_output:", feed_forward_output)
    print("feed_forward_output shape:", feed_forward_output.shape)
