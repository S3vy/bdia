#include "ann.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include "error.h"
#include <iostream>
using namespace std;

double normalRand(double mu, double sigma);
void init_weight(matrix_t* w, unsigned nneurones_prev);
void print_layer(layer_t *layer);

double normalRand(double mu, double sigma)
{
	const double epsilon = DBL_MIN;
	const double two_pi = 2.0*M_PI;
    bool generate;
    double z1;

	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	double u1, u2;
	do
	 {
	   u1 = (double) rand() / RAND_MAX;
	   u2 = (double) rand() / RAND_MAX;
	 }
	while ( u1 <= epsilon );

	double z0;
	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

void init_weight(matrix_t* w, unsigned nneurones_prev)
{
    for (int idx = 0; idx < w->columns * w->rows; idx ++)
    {
        w->m[idx] = normalRand(0, 1 / sqrt(nneurones_prev));
    }
}

ann_t * create_ann(double alpha, unsigned minibatch_size, unsigned number_of_layers, unsigned* nneurons_per_layer)
{
    ann_t * nn = (ann_t *)malloc(sizeof(ann_t));

    nn->layers = (layer_t **)malloc(number_of_layers * sizeof(layer_t *));
    nn->number_of_layers = number_of_layers;
    nn->alpha = alpha;
    nn->minibatch_size = minibatch_size;

    nn->layers[0] = create_layer(0, nneurons_per_layer[0], minibatch_size, minibatch_size);
    for (int l = 1; l < number_of_layers; l++)
    {
        nn->layers[l] = create_layer(l, nneurons_per_layer[l], nneurons_per_layer[l-1], minibatch_size);
    }

    return nn;
}

layer_t * create_layer(unsigned layer_number, unsigned number_of_neurons, unsigned nneurons_previous_layer, unsigned minibatch_size)
{
    layer_t * layer = (layer_t*) malloc(sizeof(layer_t));

    layer->number_of_neurons = number_of_neurons;
    layer->minibatch_size = minibatch_size;    
    layer->activations = alloc_matrix(number_of_neurons, minibatch_size);
    layer->z = alloc_matrix(number_of_neurons, minibatch_size);
    layer->delta = alloc_matrix(number_of_neurons, minibatch_size);
    layer->weights = alloc_matrix(number_of_neurons, nneurons_previous_layer);    
    layer->biases = alloc_matrix(number_of_neurons, 1);

    if (layer_number > 0)
    {
        init_weight(layer->weights, nneurons_previous_layer);
    }

    return layer;
}

void set_input(ann_t *nn, matrix_t* input){
    matrix_memcpy(nn->layers[0]->activations, input);
}

void print_layer(layer_t *layer)
{
    printf("-- neurons:%d, minibatch size:%d\n", layer->number_of_neurons, layer->minibatch_size);

    printf(">> Weighted inputs --\n");
    print_matrix(layer->z, true);
    printf(">> Activations --\n");
    print_matrix(layer->activations, true);
    
    printf(">> Weights --\n");
    print_matrix(layer->weights, true);
    printf(">> Biases --\n");
    print_matrix(layer->biases, true);

    printf(">> Delta --\n");
    print_matrix(layer->delta, true);
    
}

void print_nn(ann_t *nn)
{
    printf("ANN -- nlayers:%d, alpha:%lf, minibatch size: %d\n", nn->number_of_layers, nn->alpha, nn->minibatch_size);
    for (int l = 0; l < nn->number_of_layers; l++)
    {
        printf("Layer %d ", l);
        print_layer(nn->layers[l]);
    }
}

void forward(ann_t *nn, double (*activation_function)(double))
{
    // cout << "ForwardStart" << endl;
    dim3 blockDim(16, 16);
    dim3 gridDim(0, 0);

    double *d_m1;
    double *d_m2;
    double *d_res_1;
    double *d_res_2;
    double *d_res;

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *z1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *z2 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->minibatch_size);
        matrix_t *one = alloc_matrix(1, nn->minibatch_size);
        for (int idx = 0; idx < one->columns*one->rows; idx++) {
            one->m[idx] = 1.0;
        }
        // CPU version
        // matrix_dot(nn->layers[l]->weights, nn->layers[l-1]->activations, z1); // z1 <- w^l x a^(l-1)
        // matrix_dot(nn->layers[l]->biases, one, z2); // z2 <- b^l x 1

        CHECK_ERROR(cudaMalloc((void **)&d_m1, nn->layers[l]->weights->rows * nn->layers[l]->weights->columns * sizeof(double)));
        CHECK_ERROR(cudaMalloc((void **)&d_m2, nn->layers[l-1]->activations->rows * nn->layers[l-1]->activations->columns * sizeof(double)));
        CHECK_ERROR(cudaMalloc((void **)&d_res_1, z1->rows * z1->columns * sizeof(double)));

        // Copy from CPU memory to GPU
        CHECK_ERROR(cudaMemcpy(d_m1, nn->layers[l]->weights->m, nn->layers[l]->weights->rows * nn->layers[l]->weights->columns * sizeof(double), cudaMemcpyHostToDevice));
        // cout << "cudaMemcpy1" << endl;
        CHECK_ERROR(cudaMemcpy(d_m2, nn->layers[l-1]->activations->m, nn->layers[l-1]->activations->rows * nn->layers[l-1]->activations->columns * sizeof(double), cudaMemcpyHostToDevice));
        // cout << "cudaMemcpy2" << endl;

        // Launch kernel
        gridDim = dim3((nn->layers[l-1]->activations->columns + blockDim.x - 1) / blockDim.x, (nn->layers[l]->weights->rows + blockDim.y - 1) / blockDim.y);
        // gridDim = dim3((z1->columns + blockDim.x - 1) / blockDim.x, (z1->rows + blockDim.y - 1) / blockDim.y);
        matrix_dot_tile_cuda<<<gridDim, blockDim>>>(d_m1, d_m2, d_res_1, nn->layers[l]->weights->rows, nn->layers[l]->weights->columns, nn->layers[l-1]->activations->rows, nn->layers[l-1]->activations->columns); // z1 <- w^l x a^(l-1)
        // cout << "cudaKernel" << endl;

        CHECK_ERROR(cudaDeviceSynchronize());
        // cout << "cudaDeviceSynchronize" << endl;

        // Copy from GPU memory to CPU
        CHECK_ERROR(cudaMemcpy(z1->m, d_res_1, z1->rows * z1->columns * sizeof(double), cudaMemcpyDeviceToHost));
        // cout << "cudaMemcpy3" << endl;

        CHECK_ERROR(cudaFree(d_m1));
        // cout << "cudaFree1" << endl;
        CHECK_ERROR(cudaFree(d_m2));
        // cout << "cudaFree2" << endl;
        CHECK_ERROR(cudaFree(d_res_1));
        // cout << "cudaFree3" << endl;


        // Memory allocation on the GPU
        cudaMalloc((void **)&d_m1, nn->layers[l]->biases->rows * nn->layers[l]->biases->columns * sizeof(double));
        cudaMalloc((void **)&d_m2, one->rows * one->columns * sizeof(double));
        cudaMalloc((void **)&d_res_2, z2->rows * z2->columns * sizeof(double));

        // Copy from CPU memory to GPU
        cudaMemcpy(d_m1, nn->layers[l]->biases->m, nn->layers[l]->biases->rows * nn->layers[l]->biases->columns * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m2, one->m, one->rows * one->columns * sizeof(double), cudaMemcpyHostToDevice);

        // Launch kernel
        gridDim = dim3((one->columns + blockDim.x - 1) / blockDim.x, (nn->layers[l]->biases->rows + blockDim.y - 1) / blockDim.y);
        // gridDim = dim3((z2->columns + blockDim.x - 1) / blockDim.x, (z2->rows + blockDim.y - 1) / blockDim.y);
        matrix_dot_tile_cuda<<<gridDim, blockDim>>>(d_m1, d_m2, d_res_2, nn->layers[l]->biases->rows, nn->layers[l]->biases->columns, one->rows, one->columns); // z2 <- b^l x 1

        cudaDeviceSynchronize();

        // Copy from GPU memory to CPU
        cudaMemcpy(z2->m, d_res_2, z2->rows * z2->columns * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_m1);
        cudaFree(d_m2);
        cudaFree(d_res_2);

        // Sum
        matrix_sum(z1, z2, nn->layers[l]->z);
        // cudaMalloc((void **)&d_res, nn->layers[l]->biases->rows * nn->layers[l]->biases->columns * sizeof(double));
        // matrix_sum_cuda<<<gridDim, blockDim>>>(d_res_1, d_res_2, d_res, z1->rows, z1->columns, z2->rows, z2->columns); // z^l <- z1 + z2 <=> z^l <- w^l x a^(l-1) + b^l x 1      
       
        // cudaMemcpy(nn->layers[l]->z->m, d_res, nn->layers[l]->z->rows * nn->layers[l]->z->columns * sizeof(double), cudaMemcpyDeviceToHost);

        // cudaFree(d_res_1);
        // cudaFree(d_res_2);
        // cudaFree(d_res);

        matrix_function(nn->layers[l]->z, activation_function, nn->layers[l]->activations); // a^l = f(z^l)
     
        destroy_matrix(z1);
        destroy_matrix(z2);
        destroy_matrix(one);
    }

    cudaFree(d_m1);
    // cout << "cudaFreeOut1" << endl;
    cudaFree(d_m2);
    // cout << "cudaFreeOut2" << endl;
    cudaFree(d_res_1);
    cudaFree(d_res_2);
    cudaFree(d_res);
    // cout << "cudaFreeOut3" << endl;

    // cout << "ForwardEnd" << endl;
}

void backward(ann_t *nn, matrix_t *y, double (*derivative_actfunct)(double))
{
    // cout << "BackwardStart" << endl;
    unsigned L = nn->number_of_layers-1;

    matrix_t *dfzL = alloc_matrix(nn->layers[L]->number_of_neurons, nn->minibatch_size);

    matrix_minus(nn->layers[L]->activations, y, nn->layers[L]->delta);  // delta^(L) = (a^L - y)
    matrix_function(nn->layers[L]->z, derivative_actfunct, dfzL); // f'(z^(L))
    hadamard_product(nn->layers[L]->delta, dfzL, nn->layers[L]->delta); // delta^(L) = (a^L - y) o f'(z^(L))

    destroy_matrix(dfzL);

    dim3 blockDim(16, 16);
    dim3 gridDim(0, 0);

    double *d_m1;
    double *d_m2;
    double *d_res;

    for (int l = L; l > 1; l--)
    {
        matrix_t *tw, *delta_tmp, *dfz;
        tw = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->layers[l]->number_of_neurons);
        delta_tmp = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);
        dfz = alloc_matrix(nn->layers[l-1]->number_of_neurons, nn->minibatch_size);

        matrix_transpose(nn->layers[l]->weights, tw); // (w^l)T

        // CPU version        
        // matrix_dot(tw, nn->layers[l]->delta, delta_tmp); // (w^l)T x delta^l


        cudaMalloc((void **)&d_m1, tw->rows * tw->columns * sizeof(double));
        cudaMalloc((void **)&d_m2, nn->layers[l]->delta->rows * nn->layers[l]->delta->columns * sizeof(double));  
        cudaMalloc((void **)&d_res, delta_tmp->rows * delta_tmp->columns * sizeof(double));

        // Copy from CPU memory to GPU
        cudaMemcpy(d_m1, tw->m, tw->rows * tw->columns * sizeof(double), cudaMemcpyHostToDevice);
        // cout << "cudaMemcpy1" << endl;
        cudaMemcpy(d_m2, nn->layers[l]->delta->m, nn->layers[l]->delta->rows * nn->layers[l]->delta->columns * sizeof(double), cudaMemcpyHostToDevice);
        // cout << "cudaMemcpy2" << endl;

        // Launch kernel
        gridDim = dim3((nn->layers[l]->delta->columns + blockDim.x - 1) / blockDim.x, (tw->rows + blockDim.y - 1) / blockDim.y);
        matrix_dot_tile_cuda<<<gridDim, blockDim>>>(d_m1, d_m2, d_res, tw->rows, tw->columns, nn->layers[l]->delta->rows, nn->layers[l]->delta->columns);
        // cout << "cudaKernel" << endl;

        cudaDeviceSynchronize();
        // cout << "cudaDeviceSynchronize" << endl;

        // Copy from GPU memory to CPU
        cudaMemcpy(delta_tmp->m, d_res, tw->rows * nn->layers[l]->delta->columns * sizeof(double), cudaMemcpyDeviceToHost);
        // cout << "cudaMemcpy3" << endl;

        cudaFree(d_m1);
        // cout << "cudaFree1" << endl;
        cudaFree(d_m2);
        // cout << "cudaFree2" << endl;
        cudaFree(d_res);
        // cout << "cudaFree3" << endl;

        
        matrix_function(nn->layers[l-1]->z, derivative_actfunct, dfz); // f'(z^(l-1))
        hadamard_product(delta_tmp, dfz, nn->layers[l-1]->delta); // delta^(l-1) = (w^l)T x delta^l o f'(z^(l-1))

        destroy_matrix(tw);
        destroy_matrix(delta_tmp);
        destroy_matrix(dfz);
    }

    cudaFree(d_m1);
    // cout << "cudaFreeOut1" << endl;
    cudaFree(d_m2);
    // cout << "cudaFreeOut2" << endl;
    cudaFree(d_res);
    // cout << "cudaFreeOut3" << endl;

    for (int l = 1; l < nn->number_of_layers; l++)
    {
        matrix_t *w1, *ta;
        w1 = alloc_matrix(nn->layers[l]->number_of_neurons, nn->layers[l-1]->number_of_neurons);
        ta = alloc_matrix(nn->minibatch_size, nn->layers[l-1]->number_of_neurons);
        
        matrix_transpose(nn->layers[l-1]->activations, ta); // ta <- (a^(l-1))^T

        // CPU version
        // matrix_dot(nn->layers[l]->delta, ta, w1); // w1 <- delta^l x (a^(l-1))^T

        // CUDA version 1
        // Allocation de mémoire unifiée
        // cudaMallocManaged(&w1->m, w1->rows * w1->columns * sizeof(float));
        // cudaMallocManaged(&nn->layers[l]->delta->m, nn->layers[l]->delta->rows * nn->layers[l]->delta->columns * sizeof(float));
        // cudaMallocManaged(&ta->m, ta->rows * ta->columns * sizeof(float));
        // Définition de la configuration des blocks et des threads
        // dim3 threadsPerBlock(32,32);
        // dim3 numBlocks((w1->columns + threadsPerBlock.x - 1) / threadsPerBlock.x, (w1->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
        // Appel du kernel
        // matrix_dot_cuda<<<numBlocks, threadsPerBlock>>>(nn->layers[l]->delta, ta, w1);
        // Synchronisation pour s'assurer que le kernel a terminé
        // cudaDeviceSynchronize();
        // Libération de la mémoire unifiée
        // cudaFree(w1->m);

        // CUDA version 2
        // float *d_m1;
        // float *d_m2;
        // float *d_res;

        // Memory allocation on the GPU
        cudaMalloc((void **)&d_m1, nn->layers[l]->delta->rows * nn->layers[l]->delta->columns * sizeof(double));
        cudaMalloc((void **)&d_m2, ta->rows * ta->columns * sizeof(double));
        cudaMalloc((void **)&d_res, w1->rows * w1->columns * sizeof(double));

        // Copy from CPU memory to GPU
        cudaMemcpy(d_m1, nn->layers[l]->delta->m, nn->layers[l]->delta->rows * nn->layers[l]->delta->columns * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m2, ta->m, ta->rows * ta->columns * sizeof(double), cudaMemcpyHostToDevice);

        // Launch kernel
        gridDim = dim3((ta->columns + blockDim.x - 1) / blockDim.x, (nn->layers[l]->delta->rows + blockDim.y - 1) / blockDim.y);
        matrix_dot_tile_cuda<<<gridDim, blockDim>>>(d_m1, d_m2, d_res, nn->layers[l]->delta->rows, nn->layers[l]->delta->columns, ta->rows, ta->columns);

        cudaDeviceSynchronize();

        // Copy from GPU memory to CPU
        cudaMemcpy(w1->m, d_res, nn->layers[l]->delta->rows * ta->columns * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_m1);
        cudaFree(d_m2);
        cudaFree(d_res);

        // free(d_m1);
        // free(d_m2);
        // free(d_res);
        // END CUDA PART

        matrix_scalar(w1, nn->alpha / nn->minibatch_size, w1); // w1 <- alpha /m . delta^l x (a^(l-1))^T
        matrix_minus(nn->layers[l]->weights, w1, nn->layers[l]->weights); // w^l <- w^l - alpha /m . delta^l x (a^(l-1))^T

        destroy_matrix(w1);
        destroy_matrix(ta);

        matrix_t *one, *b1;
        b1 = alloc_matrix(nn->layers[l]->number_of_neurons, 1);
        one = alloc_matrix(nn->minibatch_size, 1);
        for (int idx = 0; idx < one->columns*one->rows; idx++)
            one->m[idx] = 1.0;

        // CPU version
        // matrix_dot(nn->layers[l]->delta, one, b1); // b1 <- delta^l x 1^T

        // CUDA version 1
        // Allocation de mémoire unifiée
        // cudaMallocManaged(&b1->m, b1->rows * b1->columns * sizeof(float));
        // cudaMallocManaged(&nn->layers[l]->delta->m, nn->layers[l]->delta->rows * nn->layers[l]->delta->columns * sizeof(float));
        // cudaMallocManaged(&one->m, one->rows * one->columns * sizeof(float));
        // Définition de la configuration des blocks et des threads
        // numBlocks = (b1->columns + threadsPerBlock.x - 1) / threadsPerBlock.x, (b1->rows + threadsPerBlock.y - 1) / threadsPerBlock.y;
        // Appel du kernel
        // matrix_dot_cuda<<<numBlocks, threadsPerBlock>>>(nn->layers[l]->delta, one, b1);
        // Synchronisation pour s'assurer que le kernel a terminé
        // cudaDeviceSynchronize();
        // Libération de la mémoire unifiée
        // cudaFree(b1->m);

        // CUDA version 2
        // float *d_m1;
        // float *d_m2;
        // float *d_res;

        // Memory allocation on the GPU
        cudaMalloc((void **)&d_m1, nn->layers[l]->delta->rows * nn->layers[l]->delta->columns * sizeof(double));
        cudaMalloc((void **)&d_m2, one->rows * one->columns * sizeof(double));
        cudaMalloc((void **)&d_res, b1->rows * b1->columns * sizeof(double));

        // Copy from CPU memory to GPU
        cudaMemcpy(d_m1, nn->layers[l]->delta->m, nn->layers[l]->delta->rows * nn->layers[l]->delta->columns * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_m2, one->m, one->rows * one->columns * sizeof(double), cudaMemcpyHostToDevice);

        // Launch kernel
        gridDim = dim3((one->columns + blockDim.x - 1) / blockDim.x, (nn->layers[l]->delta->rows + blockDim.y - 1) / blockDim.y);
        matrix_dot_tile_cuda<<<gridDim, blockDim>>>(d_m1, d_m2, d_res, nn->layers[l]->delta->rows, nn->layers[l]->delta->columns, one->rows, one->columns);

        cudaDeviceSynchronize();

        // Copy from GPU memory to CPU
        cudaMemcpy(b1->m, d_res, nn->layers[l]->delta->rows * one->columns * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_m1);
        cudaFree(d_m2);
        cudaFree(d_res);

        // free(d_m1);
        // free(d_m2);
        // free(d_res);
        // END CUDA PART

        matrix_scalar(b1,  nn->alpha / nn->minibatch_size, b1); // b1 <- alpha / m . delta^l x 1^T
        matrix_minus(nn->layers[l]->biases, b1, nn->layers[l]->biases); // b^l = b^l - alpha / m . delta^l x 1^T
        
        destroy_matrix(one);
        destroy_matrix(b1);
    }

    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);

    // cout << "BackwardEnd" << endl;
}