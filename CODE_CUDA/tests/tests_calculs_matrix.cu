#include "../matrix.h"
#include "../error.h"
#include "tests_calculs_matrix.h"
#include <stdlib.h>
#include <string.h>

#include <iostream>
using namespace std;

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void test_sum(matrix_t *m1, matrix_t *m2, matrix_t *expected_res)
{
    cout << "Start test_sum" << endl;
    dim3 blockDim(4, 4);
    dim3 gridDim(0, 0);

    matrix_t *res = alloc_matrix(2, 3);
    
    double *d_m1;
    double *d_m2;
    double *d_res;
    
    // Memory allocation on the GPU
    CHECK_ERROR(cudaMalloc((void **)&d_m1, 6 * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void **)&d_m2, 6 * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void **)&d_res, 6 * sizeof(double)));

    // Copy from CPU memory to GPU
    CHECK_ERROR(cudaMemcpy(d_m1, m1->m, 6 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_m2, m2->m, 6 * sizeof(double), cudaMemcpyHostToDevice));

    // Launch Kernel
    gridDim = dim3((m1->columns + blockDim.x - 1) / blockDim.x, (m1->rows + blockDim.y - 1) / blockDim.y);
    matrix_sum_cuda<<<gridDim, blockDim>>>(d_m1, d_m2, d_res, 2, 3, 2, 3); 
    cout << "cudaKernel" << endl;

    cudaDeviceSynchronize();

    // Copy from GPU memory to CPU
    cudaMemcpy(res->m, d_res, 6 * sizeof(double), cudaMemcpyDeviceToHost);

    CHECK_ERROR(cudaFree(d_m1));
    CHECK_ERROR(cudaFree(d_m2));
    CHECK_ERROR(cudaFree(d_res));

    cout << "End computations" << endl;
    cout << " " << endl; 
    cout << "Begin testing" << endl; 

    for(int idx = 0; idx < 6; idx++)
    {
        if(res->m[idx] == expected_res->m[idx])
        {
            printf("Coefficient %d is right\n", idx);
            printf("Expected %lf and got %lf\n \n", expected_res->m[idx], res->m[idx]);
        }
        else
        {
            printf("Error on coefficient %d,\n", idx);
            printf("Expected %lf and got %lf\n \n", expected_res->m[idx], res->m[idx]);
        }
    }
}

void test_dot(matrix_t *m1, matrix_t *m2, matrix_t *expected_res)
{
    cout << "Start test_dot" << endl;
    dim3 blockDim(4, 4);
    dim3 gridDim(0, 0);

    matrix_t *res = alloc_matrix(2, 2);
    
    double *d_m1;
    double *d_m2;
    double *d_res;
    
    // Memory allocation on the GPU
    CHECK_ERROR(cudaMalloc((void **)&d_m1, 6 * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void **)&d_m2, 6 * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void **)&d_res, 4 * sizeof(double)));

    // Copy from CPU memory to GPU
    CHECK_ERROR(cudaMemcpy(d_m1, m1->m, 6 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_m2, m2->m, 6 * sizeof(double), cudaMemcpyHostToDevice));

    // Launch Kernel
    gridDim = dim3((m1->columns + blockDim.x - 1) / blockDim.x, (m2->rows + blockDim.y - 1) / blockDim.y);
    matrix_dot_cuda<<<gridDim, blockDim>>>(d_m1, d_m2, d_res, 2, 3, 3, 2); 
    cout << "cudaKernel" << endl;

    cudaDeviceSynchronize();

    // Copy from GPU memory to CPU
    cudaMemcpy(res->m, d_res, 4 * sizeof(double), cudaMemcpyDeviceToHost);

    CHECK_ERROR(cudaFree(d_m1));
    CHECK_ERROR(cudaFree(d_m2));
    CHECK_ERROR(cudaFree(d_res));

    cout << "End computations" << endl;
    cout << " " << endl; 
    cout << "Begin testing" << endl; 

    for(int idx = 0; idx < 4; idx++)
    {
        if(res->m[idx] == expected_res->m[idx])
        {
            printf("Coefficient %d is right\n", idx);
            printf("Expected %lf and got %lf\n \n", expected_res->m[idx], res->m[idx]);
        }
        else
        {
            printf("Error on coefficient %d,\n", idx);
            printf("Expected %lf and got %lf\n \n", expected_res->m[idx], res->m[idx]);
        }
    }
}

void test_dot_tile(matrix_t *m1, matrix_t *m2, matrix_t *expected_res)
{
    cout << "Start test_dot_tile" << endl;
    dim3 blockDim(4, 4);
    dim3 gridDim(0, 0);

    matrix_t *res = alloc_matrix(2, 2);
    
    double *d_m1;
    double *d_m2;
    double *d_res;
    
    // Memory allocation on the GPU
    CHECK_ERROR(cudaMalloc((void **)&d_m1, 6 * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void **)&d_m2, 6 * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void **)&d_res, 4 * sizeof(double)));

    // Copy from CPU memory to GPU
    CHECK_ERROR(cudaMemcpy(d_m1, m1->m, 6 * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_m2, m2->m, 6 * sizeof(double), cudaMemcpyHostToDevice));

    // Launch Kernel
    gridDim = dim3((m1->columns + blockDim.x - 1) / blockDim.x, (m2->rows + blockDim.y - 1) / blockDim.y);
    matrix_dot_tile_cuda<<<gridDim, blockDim>>>(d_m1, d_m2, d_res, 2, 3, 3, 2); 
    cout << "cudaKernel" << endl;

    cudaDeviceSynchronize();

    // Copy from GPU memory to CPU
    cudaMemcpy(res->m, d_res, 4 * sizeof(double), cudaMemcpyDeviceToHost);

    CHECK_ERROR(cudaFree(d_m1));
    CHECK_ERROR(cudaFree(d_m2));
    CHECK_ERROR(cudaFree(d_res));

    cout << "End computations" << endl;
    cout << " " << endl; 
    cout << "Begin testing" << endl; 

    for(int idx = 0; idx < 4; idx++)
    {
        if(res->m[idx] == expected_res->m[idx])
        {
            printf("Coefficient %d is right\n", idx);
            printf("Expected %lf and got %lf\n \n", expected_res->m[idx], res->m[idx]);
        }
        else
        {
            printf("Error on coefficient %d,\n", idx);
            printf("Expected %lf and got %lf\n \n", expected_res->m[idx], res->m[idx]);
        }
    }
}

int main(int argc, char *argv[])
{
    matrix_t *m1 = alloc_matrix(2, 3);
    matrix_t *m2 = alloc_matrix(2, 3);
    matrix_t *m3 = alloc_matrix(3, 2);

    for (int idx = 0; idx < 6; idx++)
    {
        m1->m[idx] = idx%2;
        m2->m[idx] = 1;
        m3->m[idx] = idx;
    }
        
    matrix_t *res_sum = alloc_matrix(2, 3);
    matrix_t *res_dot = alloc_matrix(2, 2);

    for (int idx = 0; idx < 6; idx++)
    {
        res_sum->m[idx] = 1 + idx%2;
    }

    for (int idx = 0; idx < 4; idx++)
    {
        res_dot->m[idx] = 6 + 3*(idx%2);
    }

    test_sum(m1, m2, res_sum);

    test_dot(m2, m3, res_dot);

    test_dot_tile(m2, m3, res_dot);

    return 0;
}

