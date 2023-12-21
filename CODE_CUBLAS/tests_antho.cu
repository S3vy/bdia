#include "cublas_v2.h"
#include <stdio.h>

//#include <iostream>
//using std::cout;

void print_matrix(float *data, int rows, int cols) {
    printf("[");
    for( int row=0; row < rows; row++) {
        printf("[");
        for( int col=0; col < cols; col++) {
            printf("%f ,", data[row*cols+col]);
        }
        printf("]");
    }
    printf("]");
}

int main() {
    // allocate host vector
    /* std::vector<float> h_a = {1,2,3,4,5,6,7,8,9,10}; */
    int N = 12;
    int size = N * sizeof(int);
    float *a,*b;

    cudaMallocManaged( (void **) &a, size );
    cudaMallocManaged( (void **) &b, size );

    for (int i = 0; i < N; i++){
        a[i] = i;
    }

    // define the number or rows and the number of columns
    int m=2,n=5;

    // perform a transpose
    {
        float alpha=1;
        float *A=a;
        int lda=n;

        float beta=0;
        float *B=NULL;
        int ldb=n;

        float *C=b;
        int ldc=m;
        
        cublasHandle_t handle;
        
        cublasCreate(&handle);

        cublasStatus_t success=cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
        if ( success != CUBLAS_STATUS_SUCCESS)
            printf("\33[31mError: %d \33[0m\n", success);
        cublasDestroy(handle);
        
    }

    
    printf("origional:  ");
    print_matrix(a,m,n);
    printf("\n");

    printf("transposed: ");
    print_matrix(b,n,m);
    printf("\n");
    return 0;
}