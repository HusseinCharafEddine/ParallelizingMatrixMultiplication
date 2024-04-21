#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void matrixMultiplicationCPU(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k)
            {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
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

    struct timeval start_cpu, end_cpu;
    gettimeofday(&start_cpu, NULL);
    matrixMultiplicationCPU(a, b, c_ref, n);
    gettimeofday(&end_cpu, NULL);
    double elapsedTimeCPU = (end_cpu.tv_sec - start_cpu.tv_sec) + (end_cpu.tv_usec - start_cpu.tv_usec) / 1.0e6;
    printf("CPU execution time: %.6f seconds\n", elapsedTimeCPU);

    free(a);
    free(b);

    return 0;
}
