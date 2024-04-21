#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define TILE_SIZE 16

__global__ void matrixMultiplication(float *a, float *b, float *c, int n)
{
    __shared__ float s_a[TILE_SIZE][TILE_SIZE];
    __shared__ float s_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        int tileRow = blockIdx.y * TILE_SIZE + threadIdx.y;
        int tileCol = tile * TILE_SIZE + threadIdx.x;

        if (tileRow < n && tileCol < n)
            s_a[threadIdx.y][threadIdx.x] = a[tileRow * n + tileCol];
        else
            s_a[threadIdx.y][threadIdx.x] = 0.0f;

        tileRow = tile * TILE_SIZE + threadIdx.y;
        tileCol = blockIdx.x * TILE_SIZE + threadIdx.x;

        if (tileRow < n && tileCol < n)
            s_b[threadIdx.y][threadIdx.x] = b[tileRow * n + tileCol];
        else
            s_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += s_a[threadIdx.y][k] * s_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n)
    {
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

    dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrixMultiplication<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);
    double elapsedTimeCUDA = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1.0e6;
    printf("CUDA execution time: %.6f seconds\n", elapsedTimeCUDA);

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
