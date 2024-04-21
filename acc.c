#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void matrixMultiplication(float *a, float *b, float *c, int n)
{
#pragma acc parallel loop collapse(2) present(a[0 : n * n], b[0 : n * n], c[0 : n * n])
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

    struct timeval start_acc, end_acc;
    gettimeofday(&start_acc, NULL);

#pragma acc data copyin(a[0 : n * n], b[0 : n * n]) copyout(c[0 : n * n])
    {
        matrixMultiplication(a, b, c, n);
    }

    gettimeofday(&end_acc, NULL);
    double elapsedTimeAcc = (end_acc.tv_sec - start_acc.tv_sec) + (end_acc.tv_usec - start_acc.tv_usec) / 1.0e6;
    printf("OpenACC execution time: %.6f seconds\n", elapsedTimeAcc);

    free(a);
    free(b);
    free(c);
    free(c_ref);

    return 0;
}
