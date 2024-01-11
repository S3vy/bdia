#include "cublas_v2.h"

cublasHandle_t handle;
cublasCreate(&handle);

// Déclaration des matrices host et device
float *h_A, *h_B, *h_C;
float *d_A, *d_B, *d_C;
int m, n, k; // dimensions des matrices

// Allocation mémoire sur le host et le device, et initialisation des données

// Copie des données du host vers le device
cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

// Exécution de la multiplication de matrices sur le device avec cuBLAS
const float alpha = 1.0f;
const float beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);

// Copie des résultats du device vers le host
cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

cublasDestroy(handle);