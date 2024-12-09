import torch

# Set the random seed
torch.manual_seed(42)

# Generate a random tensor
random_tensor1 = torch.rand(3, 3)
print(random_tensor1)

# Reset the seed and generate the tensor again
torch.manual_seed(42)
random_tensor2 = torch.rand(3, 3)
print(random_tensor2)

# Check if the two tensors are the same
print(torch.equal(random_tensor1, random_tensor2))  # Output: True
