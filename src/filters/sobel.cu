#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../utils.hpp"

#define BLOCK_SIZE 16
#define SOBEL_RADIUS 1

namespace sobel
{

  __device__ const int MASK_X[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  __device__ const int MASK_Y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

  __global__ void makeConvolution(float *d_input, float *d_output, int width, int height)
  {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width)
    {
      float sumX = 0;
      float sumY = 0;

      for (int i = -SOBEL_RADIUS; i <= SOBEL_RADIUS; i++)
      {
        for (int j = -SOBEL_RADIUS; j <= SOBEL_RADIUS; j++)
        {
          int drow = row + i;
          int dcol = col + j;

          if (drow >= 0 && drow < height && dcol >= 0 && dcol < width)
          {
            sumX += d_input[drow * width + dcol] * MASK_X[i + SOBEL_RADIUS][j + SOBEL_RADIUS];
            sumY += d_input[drow * width + dcol] * MASK_Y[i + SOBEL_RADIUS][j + SOBEL_RADIUS];
          }
        }
      }
      
      d_output[row * width + col] = sqrt(sumX * sumX + sumY * sumY);
    }
  }

  void filter(float *input, float *output, int width, int height)
  {
    float *d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(float));
    cudaMalloc(&d_output, width * height * sizeof(float));
    cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    makeConvolution<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
  }
}
