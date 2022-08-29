#ifndef UTILS
#define UTILS

#include <iostream>

namespace utils
{

  inline float *read_ASCII_PGM(const char *filename, int *width, int *height)
  {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL)
    {
      printf("Cannot open file READ %s", filename);
      return NULL;
    }

    char line[256];
    fgets(line, 256, fp);
    if (line[0] != 'P' || line[1] != '2')
    {
      printf("Invalid file format");
      return NULL;
    }

    fgets(line, 256, fp);
    while (line[0] == '#')
    {
      fgets(line, 256, fp);
    }
    sscanf(line, "%d %d", width, height);

    fgets(line, 256, fp);
    int maxVal;
    sscanf(line, "%d", &maxVal);

    float *data = (float *)malloc(*width * *height * sizeof(float));
    for (int i = 0; i < *width * *height; i++)
    {
      int val;
      fscanf(fp, "%d", &val);
      data[i] = (float)val / maxVal;
    }

    fclose(fp);

    return data;
  }

  inline int write_ASCII_PGM(const char *filename, float *data, int width, int height)
  {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL)
    {
      printf("Cannot open file WRITE %s", filename);
      return 0;
    }

    fprintf(fp, "P2\n");
    fprintf(fp, "# Created by CUDA\n");
    fprintf(fp, "%d %d\n", width, height);
    fprintf(fp, "255");
    for (int i = 0; i < width * height; i++)
    {
      if (i % 12 == 0)
        fprintf(fp, "\n");
      fprintf(fp, "%f ", (data[i] * 255));
    }

    fclose(fp);

    return 1;
  }

  inline void applyFilter(int times, void (*filter)(float *, float *, int, int), float *h_input, float *h_output, int width, int height)
  {
    for (int i = 0; i < times; i++)
    {
      filter(h_input, h_output, width, height);
      filter(h_output, h_input, width, height);
      filter(h_input, h_output, width, height);
    }
  }
}

#endif