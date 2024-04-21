#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

__global__ void matrixMultiplication(float *a, float *b, float *c, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
        {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        printf("Usage: %s <matrix_A_file> <matrix_B_file> <matrix_size>\n", argv[0]);
        return 1;
    }

    char *matrixA_file = argv[1];
    char *matrixB_file = argv[2];
    int n = atoi(argv[3]);

    float *a, *b, *c, *c_ref;
    int size = n * n * sizeof(float);

    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);
    c_ref = (float *)malloc(size);

    FILE *fileA = fopen(matrixA_file, "r");
    if (fileA == NULL)
    {
        printf("Error opening file %s\n", matrixA_file);
        return 1;
    }
    for (int i = 0; i < n * n; ++i)
    {
        fscanf(fileA, "%f", &a[i]);
    }
    fclose(fileA);

    FILE *fileB = fopen(matrixB_file, "r");
    if (fileB == NULL)
    {
        printf("Error opening file %s\n", matrixB_file);
        return 1;
    }
    for (int i = 0; i < n * n; ++i)
    {
        fscanf(fileB, "%f", &b[i]);
    }
    fclose(fileB);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 dimGrid((n + 15) / 16, (n + 15) / 16);
    dim3 dimBlock(16, 16);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrixMultiplication<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);
    double elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1.0e6;
    printf("CUDA execution time: %.6f seconds\n", elapsedTime);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);
    free(c_ref);

    return 0;
}
