import torch
from Multi_Head_Attention import multi_head_attention
from torch import nn


def residual_connection_and_layernorm(x, attention_output, gamma, beta, eps):
    """
    Apply residual connection and layer normalization.

    Args:
        x: Input tensor of shape (batch_size, seq_length, d_model)
        attention_output: Output from the multi-head attention mechanism of shape (batch_size, seq_length, d_model)
        gamma: Scale parameter of shape (d_model,)
        beta: Bias parameter of shape (d_model,)
        eps: A small value to avoid division by zero
    """

    # Apply residual connenction
    residual_output = x + attention_output

    # Apply layer normalization
    mean = residual_output.mean(
        dim=-1, keepdim=True
    )  # calculate mean along the last dimension

    variance = (
        (residual_output - mean).pow(2).mean(dim=-1, keepdim=True)
    )  # calculate variance along the last dimension

    normalized = (residual_output - mean) / torch.sqrt(
        variance + eps
    )  # normalize the residual output

    layernorm_output = gamma * normalized + beta  # apply scale and bias

    return layernorm_output


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

    print("residual_layernorm_output:", residual_layernorm_output)
    print("residual_layernorm_output shape:", residual_layernorm_output.shape)
