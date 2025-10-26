/**
 * @file custom_kernel.cu
 * @brief Custom CUDA kernel for Conv2d weight gradient calculation (wrw).
 */

#include <torch/extension.h>
#include <stdio.h>

/**
 * @brief Custom CUDA kernel for calculating Conv2d weight gradients.
 * @param input Input tensor pointer.
 * @param grad_output Gradient output tensor pointer.
 * @param grad_weight Output gradient weight tensor pointer.
 * @param N Batch size.
 * @param C_in Number of input channels.
 * @param H_in Input height.
 * @param W_in Input width.
 * @param C_out Number of output channels.
 * @param H_out Output height.
 * @param W_out Output width.
 * @param K_h Kernel height.
 * @param K_w Kernel width.
 * @param stride Stride value.
 * @param padding Padding value.
 */
__global__ void my_custom_wrw_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weight,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int K_h, int K_w,
    int stride, int padding
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int kh = blockIdx.z / K_w;
    int kw = blockIdx.z % K_w;
    if (k >= C_out || c >= C_in || kh >= K_h || kw >= K_w) return;

    float sum = 0.0f;
    const int c_offset = c * H_in * W_in;
    const int k_offset = k * H_out * W_out;
    const int C_in_H_in_W_in = C_in * H_in * W_in;
    const int C_out_H_out_W_out = C_out * H_out * W_out;

    for (int n = 0; n < N; ++n) {
        const float* input_n = input + n * C_in_H_in_W_in + c_offset;
        const float* grad_output_n = grad_output + n * C_out_H_out_W_out + k_offset;

        for (int oh = 0; oh < H_out; ++oh) {
            int ih = oh * stride + kh - padding;
            if (ih < 0 || ih >= H_in) continue;

            const float* input_row = input_n + ih * W_in;
            const float* grad_row = grad_output_n + oh * W_out;

            int ow = 0;
            // Vectorized processing (4 elements at a time) when stride == 1
            if (stride == 1) {
                for (; ow + 3 < W_out; ow += 4) {
                    int iw = ow + kw - padding;
                    if (iw >= 0 && iw + 3 < W_in) {
                        float4 grad_vec = *reinterpret_cast<const float4*>(&grad_row[ow]);
                        float4 input_vec = *reinterpret_cast<const float4*>(&input_row[iw]);
                        sum += grad_vec.x * input_vec.x;
                        sum += grad_vec.y * input_vec.y;
                        sum += grad_vec.z * input_vec.z;
                        sum += grad_vec.w * input_vec.w;
                    } else {
                        // Fallback for boundary cases
                        for (int i = 0; i < 4 && ow + i < W_out; ++i) {
                            int iw_i = (ow + i) * stride + kw - padding;
                            if (iw_i >= 0 && iw_i < W_in) {
                                sum += grad_row[ow + i] * input_row[iw_i];
                            }
                        }
                    }
                }
            }

            // Handle remaining elements
            #pragma unroll 4
            for (; ow < W_out; ++ow) {
                int iw = ow * stride + kw - padding;
                if (iw >= 0 && iw < W_in) {
                    sum += grad_row[ow] * input_row[iw];
                }
            }
        }
    }

    int grad_weight_idx = k * C_in * K_h * K_w + c * K_h * K_w + kh * K_w + kw;
    grad_weight[grad_weight_idx] = sum;
}

 * @brief Launcher for the custom Conv2d weight gradient kernel.
 * @param input Input tensor.
 * @param grad_output Gradient output tensor.
 * @param K_h Kernel height.
 * @param K_w Kernel width.
 * @param stride Stride value.
 * @param padding Padding value.
 * @return Gradient weight tensor.
 */
torch::Tensor conv2d_wrw_launcher(
    torch::Tensor input,
    torch::Tensor grad_output,
    int K_h, int K_w,
    int stride, int padding
) {
    // Make sure tensors are on the GPU
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA (HIP) tensor");
    TORCH_CHECK(grad_output.is_cuda(), "Grad output tensor must be a CUDA (HIP) tensor");

    // Get tensor dimensions
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    int C_out = grad_output.size(1);
    int H_out = grad_output.size(2);
    int W_out = grad_output.size(3);

    // Create the output tensor (grad_weight)
    auto grad_weight = torch::zeros({C_out, C_in, K_h, K_w}, input.options());

    dim3 threads(8, 8, 4); // 8*8*4 = 256 threads
    dim3 blocks(
        (C_out + threads.x - 1) / threads.x,
        (C_in + threads.y - 1) / threads.y,
        (K_h * K_w + threads.z - 1) / threads.z
    );

    my_custom_wrw_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride, padding
    );

    return grad_weight;
}

/**
 * @brief Pybind11 module definition for exposing the custom kernel to Python.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "backward_weights",    // Python function name: my_custom_conv_lib.backward_weights()
        &conv2d_wrw_launcher,  // C++ function to call
        "Custom Conv2d Weight Backward (wrw) Kernel Launcher" // Docstring
    );
}
