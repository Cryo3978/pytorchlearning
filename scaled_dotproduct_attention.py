import torch
import numpy
from torch.nn.functional import softmax

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

batch_size = 16
seq_len = 32
vocab_size = 50000
model_dim = 512
num_heads = 8
d_k = model_dim // num_heads
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device = device)


def scaled_dotproduct_attention(batch_size, seq_len, vocab_size, model_dim, num_heads, d_k, input_ids):

	embedding_model = torch.rand((vocab_size, model_dim), device = device)
	pos_encoding = torch.rand((1, seq_len, model_dim), device = device)

	W_Q = torch.rand((model_dim, model_dim), device = device)
	W_K = torch.rand((model_dim, model_dim), device = device)
	W_V = torch.rand((model_dim, model_dim), device = device)

	"""
	x: [batch_size, seq_len, model_dim]
	W_Q: (model_dim, model dim)
	"""

	"""
	x: [batch_size, seq_len, model_dim]
	return [batch_size, seq_len, model_dim]
	"""

	x_embedded = embedding_model[input_ids]
	print(f'x_embedded:{x_embedded}')
	x = x_embedded +pos_encoding
	print(f'x:{x}')

	Q = x @ W_Q
	K = x @ W_K
	V = x @ W_V

	"""
	multi-head attention:

	Q: [batch_size, seq_len, model_dim]
	to Q: [batch_size, seq_len, num_heads, d_k]

	In this case:
	QK.transpose = [batch_size, seq_len, num_heads, d_k][batch_Size, seq_len, d_k, num_heads], which will get [batch_size, seq_len, num_heads, num_heads]. 

	We want: [batch_size, num_heads, seq_len, d_k][batch_size, num_heads, d_k, seq_len] = [batch_size, num_heads, seq_len, seq_len], so we do transpose first.

	softmax([batch_size, num_heads, seq_len, seq_len])/d_k**0.5 [batch_size, num_heads, seq_len, d_k] = [batch_size, num_heads, seq_len, d_k]

	[batch_size, num_heads, seq_len, d_k].transpose(1,2) = [batch_size, seq_len, num_heads, d_k]
	"""

	Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(1,2)
	K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(1,2)
	V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(1,2)

	attention = (softmax(Q @ K.transpose(-2,-1) / (d_k**0.5), dim = -1) @ V).transpose(1,2).reshape(batch_size, seq_len, -1)

	W_0 = torch.rand((model_dim, model_dim), device = device)

	attention = attention @ W_0

	return attention

print(scaled_dotproduct_attention(batch_size, seq_len, vocab_size, model_dim, num_heads, d_k, input_ids))
	
