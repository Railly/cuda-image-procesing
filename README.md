# Filters with CUDA

These are the filters that were implemented with CUDA.

1. [Gaussian Blur](#gaussian-blur)
2. [Sobel](#sobel)
3. [Canny](#canny)

# Usage

To use the filters, you need a NVIDIA GPU with CUDA support.

Then run the following command:

```bash
  nvcc -o <input-file> ../out/<output-file>
```

# Gaussian Blur

The Gaussian Blur filter is a filter that blurs the image using a Gaussian function.
