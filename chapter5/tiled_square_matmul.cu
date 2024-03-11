#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define TILE_WIDTH 16

__global__ void tiledMatmulKernel(float *M, float *N, float *P, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int by = blockIdx.y;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float pvalue = 0.0f;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph)
    {
        Mds[ty][tx] = M[row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[row * Width + col] = pvalue;
}

void tiledMatmulStub(float *M, float *N, float *P, int Width)
{
    int MatBytes = Width * Width * sizeof(float);
    float *M_d, *N_d, *P_d;

    cudaMalloc((void **)&M_d, MatBytes);
    cudaMalloc((void **)&N_d, MatBytes);
    cudaMalloc((void **)&P_d, MatBytes);
    cudaMemcpy(M_d, M, MatBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, MatBytes, cudaMemcpyHostToDevice);

    int n_tiles = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 dimsGrid(n_tiles, n_tiles, 1);
    dim3 dimsBlock(TILE_WIDTH, TILE_WIDTH, 1);
    tiledMatmulKernel<<<dimsGrid, dimsBlock>>>(M_d, N_d, P_d, Width);

    cudaMemcpy(P, P_d, MatBytes, cudaMemcpyDeviceToHost);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int main()
{
    float *M, *N, *P, *P_Expected;
    int Width = 1024;
    int MatBytes = Width * Width * sizeof(float);

    M = (float *)malloc(MatBytes);
    N = (float *)malloc(MatBytes);
    P = (float *)malloc(MatBytes);
    P_Expected = (float *)malloc(MatBytes);

    srand(time(NULL));
    for (int i = 0; i < Width; ++i)
    {
        for (int j = 0; j < Width; ++j)
        {
            M[i * Width + j] = (float)(rand() % 100);
            N[i * Width + j] = (float)(rand() % 100);
        }
    }

    for (int i = 0; i < Width; ++i)
    {
        for (int j = 0; j < Width; ++j)
        {
            for (int k = 0; k < Width; ++k)
            {
                P_Expected[i * Width + j] += M[i * Width + k] * N[k * Width + j];
            }
        }
    }

    tiledMatmulStub(M, N, P, Width);
    for (int i = 0; i < Width; ++i)
    {
        for (int j = 0; j < Width; ++j)
        {
            /* printf("P_%d,%d: %0.1f\n", i, j, P[i * Width + j]); */
            if (P[i * Width + j] != P_Expected[i * Width + j])
            {
                printf("Value P[%d,%d] was %0.1f, expected %0.1f\n", i, j, P[i * Width + j], 2 * M[i * Width + j]);
            }
        }
    }

    free(M);
    free(N);
    free(P);
    return 0;
}