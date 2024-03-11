#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define TILE_WIDTH 16

__global__ void tiledMatmulKernel(float *M, float *N, float *P, int J, int K, int L)
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
    for (int ph = 0; ph < K / TILE_WIDTH; ++ph)
    {
        if ((row < J) && (ph * TILE_WIDTH + tx < K))
        {
            Mds[ty][tx] = M[row * K + ph * TILE_WIDTH + tx];
        }
        else
        {
            Mds[ty][tx] = 0.0f;
        }
        if ((ph * TILE_WIDTH + ty < K) && (col < L))
        {
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * L + col];
        }
        else
        {
            Nds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    if ((row < K) && (col < L))
    {
        P[row * L + col] = pvalue;
    }
}

void tiledMatmulStub(float *M, float *N, float *P, int J, int K, int L)
{
    float *M_d, *N_d, *P_d;

    cudaMalloc((void **)&M_d, J * K * sizeof(float));
    cudaMalloc((void **)&N_d, K * L * sizeof(float));
    cudaMalloc((void **)&P_d, J * L * sizeof(float));
    cudaMemcpy(M_d, M, J * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, K * L * sizeof(float), cudaMemcpyHostToDevice);

    int n_tiles_y = (J + TILE_WIDTH - 1) / TILE_WIDTH;
    int n_tiles_x = (L + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 dimsGrid(n_tiles_x, n_tiles_y, 1);
    dim3 dimsBlock(TILE_WIDTH, TILE_WIDTH, 1);
    tiledMatmulKernel<<<dimsGrid, dimsBlock>>>(M_d, N_d, P_d, J, K, L);

    cudaMemcpy(P, P_d, J * L * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int main()
{
    float *M, *N, *P, *P_Expected;
    int J = 128;
    int K = 256;
    int L = 512;

    M = (float *)malloc(J * K * sizeof(float));
    N = (float *)malloc(K * L * sizeof(float));
    P = (float *)malloc(J * L * sizeof(float));
    P_Expected = (float *)malloc(J * L * sizeof(float));

    srand(time(NULL));
    for (int k = 0; k < K; ++k)
    {
        for (int j = 0; j < J; ++j)
        {
            M[j * K + k] = (float)(rand() % 100);
        }
        for (int l = 0; l < L; ++l)
        {
            N[k * L + l] = (float)(rand() % 100);
        }
    }

    for (int j = 0; j < J; ++j)
    {
        for (int l = 0; l < L; ++l)
        {
            float pvalue = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                pvalue += M[j * K + k] * N[k * L + l];
            }
            P_Expected[j * L + l] = pvalue;
        }
    }

    tiledMatmulStub(M, N, P, J, K, L);
    for (int j = 0; j < J; ++j)
    {
        for (int l = 0; l < L; ++l)
        {
            if (P[j * L + l] != P_Expected[j * L + l])
            {
                printf("Value P[%d,%d] was %0.1f, expected %0.1f\n", j, l, P[j * L + l], P_Expected[j * L + l]);
            }
        }
    }

    free(M);
    free(N);
    free(P);
    return 0;
}