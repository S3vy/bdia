#include "cublas_v2.h"
#include <vector>
#include <iostream>
using std::cout;

void print_matrix(float *data, int rows, int cols) {
    cout << "[";
    for( int row=0; row < rows; row++) {
        cout << "[";
        for( int col=0; col < cols; col++) {
            cout << data[row*cols+col] << ",";
        }
        cout << "]";
    }
    cout << "]";
}

int main() {
    // allocate host vector
    std::vector<float> h_a = {1,2,3,4,5,6,7,8,9,10};
    int nbytes=h_a.size()*sizeof(*h_a.data());
    std::vector<float> h_b(h_a.size());

    // define the number or rows and the number of columns
    int m=2,n=5;

    // allocate device vectors
    float *d_a, *d_b;
    cudaMalloc(&d_a, nbytes);
    cudaMalloc(&d_b, nbytes);

    // copy host vector to device
    cudaMemcpy(d_a,h_a.data(), nbytes, cudaMemcpyHostToDevice);

    // perform a transpose
    {

        float alpha=1;
        float *A=d_a;
        int lda=n;

        float beta=0;
        float *B=NULL;
        int ldb=n;

        float *C=d_b;
        int ldc=m;

        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasStatus_t success=cublasSgeam( handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A, lda, &beta, B, ldb, C, ldc);
        if ( success != CUBLAS_STATUS_SUCCESS)
            cout << "\33[31mError: " << success << "\33[0m\n";
        cublasDestroy(handle);
    }

    // copy back to host
    cudaMemcpy(h_b.data(),d_b,nbytes,cudaMemcpyDeviceToHost);

    cout << "origional:  ";
    print_matrix(h_a.data(),m,n);
    cout << "\n";

    cout << "transposed: ";
    print_matrix(h_b.data(),n,m);
    cout << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    return 0;
}