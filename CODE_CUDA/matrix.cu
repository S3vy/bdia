#include "matrix.h"
#include "error.h"
#include <stdlib.h>
#include <string.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
# define tileSize 32

matrix_t * alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t * res = (matrix_t*) malloc( sizeof(matrix_t) );
    res->m = (double *) calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    //printf("free %p %p\n", m, m->m);
    free(m->m);
    free(m);
}

void print_matrix(matrix_t *m, bool is_short){
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row ++)
    {
        for (int col = 0; col < lim_col; col ++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns) printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows) printf("...\n");
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)   &&
             (m1->columns == res->columns)  &&
             (m1->rows == m2->rows)         &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
            res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    { 
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->columns)  &&
             (m1->columns == res->columns) &&
             (m1->rows == m2->rows)        &&
             (m1->rows == res->rows));
             
    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    for (int row = 0; row < m1->rows; row ++)
    {
        for (int col = 0; col < m2->columns; col ++)
        {
            int idx = col + row * m2->columns;
            double var = 0.0;

            for (int ii = 0; ii < m1->columns; ii++)
            {
                var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
            }

            res->m[idx] = var;
        }
    }
}

__global__ void hadamard_product_cuda
(   double *m1, double *m2, double *res,
    int numM1Rows, int numM1Columns,
    int numM2Rows, int numM2Columns
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    assert ( (numM1Rows == numM2Rows)  &&
             (numM1Columns == numM1Columns));
    
    if (row < numM1Rows && col < numM1Columns) {
        res[col + row * numM1Columns] = m1[col + row * numM1Columns] * m2[col + row * numM1Columns];
    }
}

__global__ void matrix_sum_cuda
(   double *m1, double *m2, double *res,
    int numM1Rows, int numM1Columns,
    int numM2Rows, int numM2Columns
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    assert ( (numM1Rows == numM2Rows)  &&
             (numM1Columns == numM1Columns));
    
    if (row < numM1Rows && col < numM1Columns) {
        res[col + row * numM1Columns] = m1[col + row * numM1Columns] + m2[col + row * numM1Columns];
    }
}

__global__ void matrix_minus_cuda
(   double *m1, double *m2, double *res,
    int numM1Rows, int numM1Columns,
    int numM2Rows, int numM2Columns
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    assert ( (numM1Rows == numM2Rows)  &&
             (numM1Columns == numM1Columns));
    
    if (row < numM1Rows && col < numM1Columns) {
        res[col + row * numM1Columns] = m1[col + row * numM1Columns] - m2[col + row * numM1Columns];
    }
}

__global__ void matrix_dot_cuda
(   double *m1, double *m2, double *res,
    int numM1Rows, int numM1Columns,
    int numM2Rows, int numM2Columns
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    assert (numM1Columns == numM2Rows) ;


    if (row < numM1Rows && col < numM2Columns) {
        for (int i = 0; i < numM1Columns; i++) {
            sum += m1[i + row * numM1Columns] * m2[col + i * numM2Columns];
        }

        res[col + row * numM2Columns] = sum;
    }
}

__global__ void matrix_dot_tile_cuda
(
    double *m1, double *m2, double *res,
    int numM1Rows, int numM1Columns,
    int numM2Rows, int numM2Columns
)
{
    // Defining shared memory workspaces
    __shared__ float ds_m1 [ tileSize ][ tileSize ];
    __shared__ float ds_m2 [ tileSize ][ tileSize ];

    // Redefinition of blockIdx.x, blockIdx.y,
    // threadIdx.x and threadIdx.y to simplify writing
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Definition of row and col
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    // intermediate variable for calculating the sum of products
    float pvalue = 0.0;

    // Loop over the m1 and m2 tiles required to compute the P element
    for (int p = 0; p < numM1Columns / tileSize + 1 ; p ++) {
        // collaborative loading of M and N tiles into shared memory
        if ( row < numM1Rows && p * tileSize + tx < numM1Columns )
        { 
            ds_m1 [ty][tx] = m1[row * numM1Columns + p * tileSize + tx]; 
        }
        else
        { 
            ds_m1 [ty][tx] = 0.0; 
        }
        if (p * tileSize + ty < numM2Rows && col < numM2Columns )
        {
            ds_m2 [ty][tx] = m2[(p * tileSize + ty) * numM2Columns + col]; 
        }
        else
        { 
            ds_m2 [ty][tx] = 0.0;   
        }

        __syncthreads();

        if ( row < numM1Rows && col < numM2Columns )
        {
            for (int k = 0; k < tileSize ; k ++)
            {
                pvalue += ds_m1 [ty][k] * ds_m2 [k][tx];
            }
        }

        __syncthreads();

    }

    if ( row < numM1Rows && col < numM2Columns )
        { res[ row * numM2Columns + col ] = pvalue ; }

}

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert ( (m1->columns == res->columns) &&             
             (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx ++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert ( (m1->columns == res->rows) &&             
             (m1->rows == res->columns));
    
    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col ++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert ( (m1->rows == res->rows) &&             
             (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns*m1->rows; idx ++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert ( (dest->rows == src->rows)      &&             
             (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));     
}

