
#include <cstdio>
#include <driver_types.h>
struct CMatrix{
    int rows;
    int cols;
    const float *data;
    int len;
};

__global__ void matrixMul(float *result, const float *matrix_a, const float *matrix_b,  int rowsA, int colsA, int colsB){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < rowsA && col < colsB){
        float res = 0.0f;
        for(int i = 0; i < colsA; i++){
            res += matrix_a[row * colsA + i] * matrix_b[i * colsB + col];
        }
        result[row * colsB + col] = res;
    }
}
__global__ void matrixAdd(float *result, const float *matrix_a, const float *matrix_b, int rowsA, int colsA){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rowsA && col < colsA){
        float res = 0.0f;
        res += matrix_a[row * colsA + col] + matrix_b[row * colsA + col];
        result[row * colsA + col] = res;
    }
}
extern "C" {
    void cuda_matrix_mul(float *result, const CMatrix *matrix_a, const CMatrix *matrix_b){
        //Transfer to device mem :)
        float *result_dev, *matrix_a_dev, *matrix_b_dev;
        cudaMalloc((void**)&result_dev, matrix_a->rows * matrix_b->cols * sizeof(float));

        cudaMalloc((void**)&matrix_a_dev, matrix_a->len*sizeof(float));
        cudaMalloc((void**)&matrix_b_dev, matrix_b->len*sizeof(float));

        //Copy matrix A and B to A-device and B-device memory allocations
        cudaMemcpy(matrix_a_dev, matrix_a->data, matrix_a->len * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix_b_dev, matrix_b->data, matrix_b->len * sizeof(float), cudaMemcpyHostToDevice);

        //Split our grids and block sizes into blocks of 16x16
        dim3 blockSize(16,16);
        dim3 gridSize((matrix_b->cols + blockSize.x - 1) / blockSize.x, (matrix_a->rows + blockSize.y - 1) / blockSize.y);
        //Call cuda function!
        matrixMul<<<gridSize, blockSize>>>(result_dev, matrix_a_dev, matrix_b_dev, matrix_a->rows, matrix_a->cols, matrix_b->cols); 

        cudaError_t err = cudaMemcpy(result, result_dev, matrix_a->rows * matrix_b->cols * sizeof(float), cudaMemcpyDeviceToHost);
        if(err != cudaSuccess){
            printf("Big ass cuda error: %s\n", cudaGetErrorString(err));
        }
        //Memory management :D
        cudaFree(result_dev);
        cudaFree(matrix_a_dev);
        cudaFree(matrix_b_dev);
    }
    void cuda_matrix_add(float *result, const CMatrix *matrix_a, const CMatrix *matrix_b){
        //Transfer to device mem :)
        float *result_dev, *matrix_a_dev, *matrix_b_dev;
        cudaMalloc((void**)&result_dev, matrix_a->len * sizeof(float));

        cudaMalloc((void**)&matrix_a_dev, matrix_a->len*sizeof(float));
        cudaMalloc((void**)&matrix_b_dev, matrix_b->len*sizeof(float));

        //Copy matrix A and B to A-device and B-device memory allocations
        cudaMemcpy(matrix_a_dev, matrix_a->data, matrix_a->len * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(matrix_b_dev, matrix_b->data, matrix_b->len * sizeof(float), cudaMemcpyHostToDevice);

        //Split our grids and block sizes into blocks of 16x16
        dim3 blockSize(16,16);
        dim3 gridSize((matrix_b->cols + blockSize.x - 1) / blockSize.x, (matrix_a->rows + blockSize.y - 1) / blockSize.y);
        //Call cuda function!
        matrixAdd<<<gridSize, blockSize>>>(result_dev, matrix_a_dev, matrix_b_dev, matrix_a->rows, matrix_a->cols); 

        cudaMemcpy(result, result_dev, matrix_a->len * sizeof(float), cudaMemcpyDeviceToHost);
        //Memory management :D
        cudaFree(result_dev);
        cudaFree(matrix_a_dev);
        cudaFree(matrix_b_dev);
    }
}
/*
int main() {
    //Test matrix math
    int rowsA = 3;
    int colsA = 4;
    int rowsB = 4; // Should match colsA for multiplication
    int colsB = 2;

    float matrixA[rowsA * colsA];
    float matrixB[rowsB * colsB];

    for (int i = 0; i < rowsA * colsA; ++i) {
        matrixA[i] = (float)(rand() % 10); 
        printf("%f ", matrixA[i]);
    }
    printf("\n");
    for (int i = 0; i < rowsB * colsB; ++i) {
        matrixB[i] = (float)(rand() % 10); 
        printf("%f ", matrixB[i]);
    }

    struct CMatrix matA = { rowsA, colsA, matrixA, rowsA * colsA };
    struct CMatrix matB = { rowsB, colsB, matrixB, rowsB * colsB };

    int rowsResult = rowsA;
    int colsResult = colsB;

    float resultMatrix[rowsResult * colsResult];

    struct CMatrix matResult = { rowsResult, colsResult, resultMatrix, rowsResult * colsResult };

    cuda_matrix_mul(resultMatrix, &matA, &matB);

    printf("Result Matrix:\n");
    for (int i = 0; i < rowsResult; ++i) {
        for (int j = 0; j < colsResult; ++j) {
            printf("%.2f ", resultMatrix[i * colsResult + j]);
        }
        printf("\n");
    }

    return 0;
}*/
