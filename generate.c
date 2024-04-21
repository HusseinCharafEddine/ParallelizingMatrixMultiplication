#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to generate a random integer between min and max
int randomInt(int min, int max)
{
    return min + rand() % (max - min + 1);
}

// Function to generate and write a random matrix to a file
void generateRandomMatrixFile(const char *filename, int rows, int cols, int minVal, int maxVal)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file %s\n", filename);
        return;
    }

    srand(time(NULL)); // Seed the random number generator

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            int value = randomInt(minVal, maxVal);
            fprintf(file, "%d ", value);
        }
        fprintf(file, "\n"); // Newline after each row
    }

    fclose(file);
}

int main()
{
    int rows = 2048;   // Number of rows
    int cols = 2048;   // Number of columns
    int minVal = 1;    // Minimum random value
    int maxVal = 1000; // Maximum random value

    generateRandomMatrixFile("matrixA.txt", rows, cols, minVal, maxVal);
    generateRandomMatrixFile("matrixB.txt", rows, cols, minVal, maxVal);

    printf("Random matrices generated and saved to 'matrixA.txt' and 'matrixB.txt'.\n");

    return 0;
}