import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k 

    def forward(self, Q, K, V, mask):
        # shape: (batch_Size x num_heads x problem_size x d_k(d_v))
        d_k = self.d_k
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / \
                     math.sqrt(d_k) # batch_Size x num_heads x problem_size_Q x problem_size_K
        if mask is None:
            mask = torch.zeros_like(attn_score).bool() 
        else:
            mask = mask.unsqueeze(1).repeat(1, Q.size(1), 1, 1)
        attn_score[mask] = -1e9 

        attn_dist = F.softmax(attn_score, dim=-1) 
        output = torch.matmul(attn_dist, V)  # shape : (batch_Size x num_heads x problem_size x d_v) convex combination
        return output, attn_dist

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, is_encoder=True):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads 
        self.d_k = d_k 
        self.d_v = d_v 
        self.multihead_combine = nn.Linear(d_model, d_model, bias = False)
        self.attention = ScaledDotProductAttention(d_k)

    def forward(self, Q, K, V, mask):
        # shape of Q,K,V : (batch_size x problem_size x embedding_dim)
        batchSize, seqLen_Q, seqLen_K = Q.size(0), Q.size(1), K.size(1)

        Q = Q.view(batchSize, seqLen_Q, self.n_heads, self.d_k)
        K = K.view(batchSize, seqLen_K, self.n_heads, self.d_k)
        V = V.view(batchSize, seqLen_K, self.n_heads, self.d_v)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  # shape: (batch_size x num_heads x problem_size x d_k)
        output, attn_dist = self.attention(Q, K, V, mask) # shape: (batch_size x num_heads x problem_size x d_k)

        output = output.transpose(1, 2).contiguous()  # shape : (batch_size x problem_size x num_heads x d_k)
        output = output.view(batchSize, seqLen_Q, -1)  # shape : (batch_size x problem_size x num_heads x embedding_dim)
        output = self.multihead_combine(output)
        return output, attn_dist
    
class FeedForward_layer(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward_layer, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        output = self.W_2(F.relu(self.W_1(x)))
        return output   
    
class Add_And_Normalization_layer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.BN = nn.BatchNorm1d(d_model)

    def forward(self, input1, input2):
        # shape: (batch_size, problem_size, embedding_dim)
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.BN(transposed)
        normalized = normalized.transpose(1, 2)
        return normalized