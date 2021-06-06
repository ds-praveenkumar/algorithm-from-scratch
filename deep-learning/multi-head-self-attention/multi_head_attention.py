
import torch
import torch.nn.functional as F  
import math

print("torch version ", torch.__version__)

def scaled_dot_product( query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
    """ computes scaled dot product for the provided inputs 
        ARGS:
        query:
        key:
        value:
    """
    key_dim = key.size()[-1]
    atten_logits = torch.matmul( query, key.transpose(-2,-1) )
    atten_logits = atten_logits / math.sqrt( key_dim)
    if mask is not None:
        atten_logits = atten_logits.masked_fill(mask == 0, 9e15)
    attention = F.softmax(atten_logits, dim=-1)
    values = torch.matmul( attention, value)
    return attention, values
        

if __name__ == "__main__":
    query = torch.rand(3,2)
    key = torch.rand(3,2)
    value = torch.rand(3,2)
    attention, values = scaled_dot_product( query=query,    
                        key=key,
                        value=value
            )
    print( "query:\n", query)
    print( "key:\n", key)
    print( "value:\n", value)
    print("attention:\n", attention)
    print("values:\n", values)

    

