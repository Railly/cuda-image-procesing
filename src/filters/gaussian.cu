#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../utils.hpp"

#define GAUSSIAN_KERNEL_SIZE 5
#define GAUSSIAN_KERNEL_RADIUS 2

#define BLOCK_SIZE 16

namespace gaussian
{

  __device__ const float gaussian_kernel[GAUSSIAN_KERNEL_SIZE][GAUSSIAN_KERNEL_SIZE] = {
      {1.0f / 256, 4.0f / 256, 6.0f / 256, 4.0f / 256, 1.0f / 256},
      {4.0f / 256, 16.0f / 256, 24.0f / 256, 16.0f / 256, 4.0f / 256},
      {6.0f / 256, 24.0f / 256, 36.0f / 256, 24.0f / 256, 6.0f / 256},
      {4.0f / 256, 16.0f / 256, 24.0f / 256, 16.0f / 256, 4.0f / 256},
      {1.0f / 256, 4.0f / 256, 6.0f / 256, 4.0f / 256, 1.0f / 256}};

  __global__ void makeConvolution(float *d_input, float *d_output, int width, int height)
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
      return;

    float sum = 0.0f;
    for (int i = -GAUSSIAN_KERNEL_RADIUS; i <= GAUSSIAN_KERNEL_RADIUS; i++)
    {
      for (int j = -GAUSSIAN_KERNEL_RADIUS; j <= GAUSSIAN_KERNEL_RADIUS; j++)
      {
        int x1 = x + i;
        int y1 = y + j;
        if (x1 < 0 || x1 >= width || y1 < 0 || y1 >= height)
          continue;
        sum += d_input[y1 * width + x1] * gaussian_kernel[GAUSSIAN_KERNEL_RADIUS + i][GAUSSIAN_KERNEL_RADIUS + j];
      }
    }
    d_output[y * width + x] = sum;
  }

  void filter(float *h_input, float *h_output, int width, int height)
  {
    float *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));
    cudaMemcpy(d_input, h_input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    makeConvolution<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
  }
}