#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    // Initialiser cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Définir la taille des matrices
    int m = 3;
    int n = 3;
    int k = 3;

    // Allouer et initialiser les matrices sur le CPU
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(m * k * sizeof(float));
    h_B = (float*)malloc(k * n * sizeof(float));
    h_C = (float*)malloc(m * n * sizeof(float));

    // Initialiser les matrices avec des valeurs quelconques

    // Allouer de la mémoire sur le GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    // Copier les données du CPU vers le GPU
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Effectuer la multiplication de matrices sur le GPU avec cuBLAS
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);

    // Copier le résultat du GPU vers le CPU
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Afficher le résultat
    printf("Résultat de la multiplication de matrices :\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", h_C[i * n + j]);
        }
        printf("\n");
    }

    // Libérer la mémoire
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Détruire le gestionnaire cuBLAS
    cublasDestroy(handle);

    return 0;
}
