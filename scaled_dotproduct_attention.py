import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 32
seq_len = 16
vocab_size = 50000
model_dim = 512
heads = 8
d_k = model_dim // heads

input_ids = torch.randint(0, vocab_size,(batch_size, seq_len), device = device)
embedding_model = torch.rand((vocab_size, model_dim), device = device)
pos_encoding = 

def scaled_dotproduct_attention(input_ids):
    """
    input_ids: [batch_size, seq_len]
    embedding_model: [vocab_size, model_dim]
    x_embedded: [batch_size, seq_len, model_dim]
    
    """
    
    x_embedded = embedding_model[input_ids]
    
    x = x_embedded + pos_encoding