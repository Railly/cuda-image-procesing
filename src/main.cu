#include "utils.hpp"
#include "filters/gaussian.cu"
#include "filters/sharpen.cu"
#include "filters/sobel.cu"

int main()
{
  int width, height;
  float *h_input = utils::read_ASCII_PGM("lena.ascii.pgm", &width, &height);
  float *h_input2 = utils::read_ASCII_PGM("lena.ascii.pgm", &width, &height);
  float *h_input3 = utils::read_ASCII_PGM("lena.ascii.pgm", &width, &height);
  float *h_output = (float *)malloc(width * height * sizeof(float));
  float *h_output2 = (float *)malloc(width * height * sizeof(float));
  float *h_output3 = (float *)malloc(width * height * sizeof(float));

  /* Here we're making only one convolution per filter, but you can make as many as you want
    * You can do it as follows:
    * utils::applyFilter(N, gaussian::filter, h_input, h_output, width, height);
    * where N is the number of times we want to apply the filter.
    */
  gaussian::filter(h_input, h_output, width, height);
  utils::write_ASCII_PGM("lena_gaussian.ascii.pgm", h_output, width, height);
  printf("Gaussian Filter Done\n");

  sharpen::filter(h_input2, h_output2, width, height);
  printf("Sharpen Filter Done\n");
  utils::write_ASCII_PGM("lena_sharpened.ascii.pgm", h_output2, width, height);

  sobel::filter(h_input3, h_output3, width, height);
  printf("Sobel Filter Done\n");
  utils::write_ASCII_PGM("lena_sobel.ascii.pgm", h_output3, width, height);
}