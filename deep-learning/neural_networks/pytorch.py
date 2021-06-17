import torch
import torch.nn as nn



print("Using torch", torch.__version__ )

#set seed
torch.manual_seed(42)

#check for GPU
gpu_avail = torch.cuda.is_available()
print("Is the GPU available? %s" % str(gpu_avail))

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

## create tensor
tensor = torch.Tensor(2,3,4)
print(tensor)

one_array = torch.ones(2,3,4)
print("one array:\n",one_array)

zeros_array = torch.zeros(2,3,4)
print("zeros array:\n",zeros_array)

rand_array = torch.rand(2,3,4)
print("rand_array:\n",rand_array)

randn_array = torch.randn( 2,3,4)
print("randn:\n",randn_array)

arange_array = torch.arange(20, dtype=torch.int8, device='cpu')
print("arange: \n", arange_array)

# tensor to numpy
pt_to_numpy = tensor.numpy()
print( "tensor to numpy:\n", pt_to_numpy)

# numpy to pytorch
numpy_to_pt = torch.from_numpy(pt_to_numpy)
print("numpy to tensor: ", numpy_to_pt)

# reshape
reshaped = arange_array.view(-1,5)
print("reshaped:\n", reshaped)

# transpose
transpose = reshaped.permute(1,0)
print( "transose:\n", transpose)


# simple nn
class NN( nn.Module):
    def __init__(self, num_inputs, hidden_layer, num_outputs) -> None:
        super().__init__()
        self.l1 = nn.Linear( num_inputs, hidden_layer)
        self.actf = nn.ReLU()
        self.l2 = nn.Linear( num_inputs, hidden_layer)
        self.output = nn.Linear(num_outputs, hidden_layer)

    def forward(self, x):
        x = self.l1(x) 
        x = self.actf(x)
        x = self.l2(x)
        x =  self.actf()
        x = self.output(x)
        return x

model = NN(2, 4, 1)
print(model)

for name, param in model.named_parameters():
    print("Parameter %s, shape %s" % (name, str(param.shape)))

