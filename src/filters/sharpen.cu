#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../utils.hpp"

#define SHARPEN_KERNEL_SIZE 3
#define SHARPEN_KERNEL_RADIUS 1
#define SHARPEN_KERNEL_LENGTH 9

#define BLOCK_SIZE 16

namespace sharpen
{
  __device__ float d_sharpen_kernel[SHARPEN_KERNEL_SIZE][SHARPEN_KERNEL_SIZE] = {
      {0, -1, 0},
      {-1, 5, -1},
      {0, -1, 0}};

  __global__ void sharpen_kernel(float *d_input, float *d_output, int width, int height)
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
      return;

    float sum = 0;
    for (int i = -SHARPEN_KERNEL_RADIUS; i <= SHARPEN_KERNEL_RADIUS; i++)
    {
      for (int j = -SHARPEN_KERNEL_RADIUS; j <= SHARPEN_KERNEL_RADIUS; j++)
      {
        int x_i = x + i;
        int y_j = y + j;

        if (x_i < 0 || x_i >= width || y_j < 0 || y_j >= height)
          continue;
        sum += d_input[y_j * width + x_i] * d_sharpen_kernel[i + SHARPEN_KERNEL_RADIUS][j + SHARPEN_KERNEL_RADIUS];
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
    dim3 grid(ceil((float)width / BLOCK_SIZE), ceil((float)height / BLOCK_SIZE));

    sharpen_kernel<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
  }
}
