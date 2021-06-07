import torch.nn as nn
import torch
from .scaled_dot_product import scaled_dot_product
class MultiHeadAttention(nn.Module):

    def __init__( self, input_dim, embedding_dim, num_heads):
        assert embedding_dim % num_heads == 0, "Embedding dimention must be 0 modulo of number of heads "

        self.empedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv_projection = nn.Linear( input_dim, 3 * embedding_dim)
        self.output_project = nn.Linear( embedding_dim, embedding_dim)

    def _reset_parameter( self ):
        nn.init.xavier_uniform_( self..qkv_projection.weight)
        self.qkv_projection.bias.data.fill_(0)
        nn.init.xavier_uniform_( self.output_project.weight)
        self.output_project.bias.data.fill_(0)

    def forword( self, x, mask=None, return_sequence=False):
        batch_size, seq_length, embedding_dim = x.size()
        qkv = self.qkv_projection(x)

        # seperate qkv from iinear output 
        qkv = qkv.reshape( batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        # Determine value outputs
        attention, values = scaled_dot_product(q,k,v, mask=mask)
        values =    values.permute(0, 2, 1, 3)
        values = values.reshape( batch_size, seq_length, embedding_dim )
        output = self.output_project( values)

        if return_sequence:
            return attention, output
        else:
            output




