import torch

# 1d tensors
n=2
a = torch.randn(n)
b = torch.randn(n)
print( a, b )
print( torch.matmul(a, b))
print( 'element wise: ', a * b)

m=2
k=4
j=6
a=torch.randn(m,k)
b = torch.randn( k,j)
print(a,b )
print( torch.matmul(a, b))

