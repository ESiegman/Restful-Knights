##
# @file run_custom.py
# @brief Example usage and test for custom CUDA Conv2d kernel.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# --- Step 1: Import your compiled C++ extension ---
try:
    # This name must match 'name' in setup.py
    import my_custom_conv_lib
except ImportError:
    print("="*50)
    print("ERROR: Could not import C++ module 'my_custom_conv_lib'.")
    print("Please run: python3 setup.py install")
    print("This will compile your custom_kernel.cu file.")
    print("="*50)
    exit()


##
# @class CustomConvFunction
# @brief PyTorch autograd Function to override Conv2d backward with custom kernel.
class CustomConvFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding):
        """
        @brief Forward pass using standard PyTorch conv2d.
        @param ctx Autograd context.
        @param input Input tensor.
        @param weight Weight tensor.
        @param bias Bias tensor.
        @param stride Stride value.
        @param padding Padding value.
        @return Output tensor.
        """
        output = F.conv2d(input, weight, bias, stride, padding)
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        @brief Backward pass using custom C++ kernel for grad_weight.
        @param ctx Autograd context.
        @param grad_output Gradient output tensor.
        @return Gradients for input, weight, bias, stride, padding.
        """
        input, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        grad_input = grad_weight = grad_bias = None

        print("\n[Python] >>> Intercepting autograd. Calling CUSTOM C++ 'wrw' kernel! <<<\n")
        K_h, K_w = weight.shape[2:]
        grad_weight = my_custom_conv_lib.backward_weights(
            input, grad_output, K_h, K_w, stride, padding
        )

        grad_input = torch.nn.grad.conv2d_input(
            input.shape, weight, grad_output, stride, padding
        )

        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=[0, 2, 3])

        return grad_input, grad_weight, grad_bias, None, None

##
# @class MyCustomConv
# @brief Convenience nn.Module wrapper for the custom convolution.
class MyCustomConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        @brief Constructor for MyCustomConv.
        @param in_channels Number of input channels.
        @param out_channels Number of output channels.
        @param kernel_size Kernel size.
        @param stride Stride value.
        @param padding Padding value.
        """
        super().__init__()
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        """
        @brief Forward pass using CustomConvFunction.
        @param x Input tensor.
        @return Output tensor.
        """
        return CustomConvFunction.apply(x, self.weight, self.bias, self.stride[0], self.padding[0])

# --- Step 4: Use it! ---
print("Running test with custom convolution...")

# Move to GPU (PyTorch on ROCm uses the 'cuda' device name)
device = 'cuda'

# Replace nn.Conv2d(3, 16, 3) with our module
model = MyCustomConv(3, 16, kernel_size=3, stride=1, padding=1).to(device)

dummy_input = torch.randn(4, 3, 32, 32, device=device)
labels = torch.randint(0, 16, (4,), device=device)

# Standard training loop
output = model(dummy_input)
loss = output.sum() # Dummy loss

print(f"Loss: {loss.item()}")

# --- This is the magic moment ---
# When you call .backward(), it will call YOUR C++ kernel!
print("Calling loss.backward()...")
loss.backward()

print("\nBackward pass complete.")
print("Check grad on weight tensor (should have '1.2345' at index 0):")
print(model.weight.grad[0, 0, 0, 0])
print("SUCCESS! Your custom kernel ran.")
