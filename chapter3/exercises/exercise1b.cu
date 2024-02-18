/* Write a kernel that has one thread to produce each output matrix col */
#include <stdio.h>

__global__ void MatmulColthreadKernel(float *M, float *N, float *P, int imax, int kmax, int jmax)
{
    /* ik, kj -> ij */
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < jmax)
    {
        for (int i = 0; i < imax; ++i)
        {
            float pValue = 0;
            for (int k = 0; k < kmax; ++k)
            {
                pValue += M[i * kmax + k] * N[k * jmax + j];
            }
            P[i * jmax + j] = pValue;
        }
    }
}

void MatmulColthreadStub(float *M, float *N, float *P, int imax, int kmax, int jmax)
{
    int msize = imax * kmax * sizeof(float);
    int nsize = kmax * jmax * sizeof(float);
    int psize = imax * jmax * sizeof(float);
    float *M_d, *N_d, *P_d;

    cudaMalloc((void **)&M_d, msize);
    cudaMalloc((void **)&N_d, nsize);
    cudaMalloc((void **)&P_d, psize);

    cudaMemcpy(M_d, M, msize, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, nsize, cudaMemcpyHostToDevice);

    int block_sz = 256;
    dim3 dimsGrid((jmax + (block_sz - 1)) / block_sz, 1, 1);
    dim3 dimsBlock(block_sz, 1, 1);
    MatmulColthreadKernel<<<dimsGrid, dimsBlock>>>(M_d, N_d, P_d, imax, kmax, jmax);

    cudaMemcpy(P, P_d, psize, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}

int main()
{
    float *M, *N, *P;
    int imax = 2;
    int kmax = 2;
    int jmax = 2;

    M = (float *)malloc(imax * kmax * sizeof(float));
    N = (float *)malloc(kmax * jmax * sizeof(float));
    P = (float *)malloc(imax * jmax * sizeof(float));

    M[0] = 1.0;
    M[1] = 0.0;
    M[2] = 0.0;
    M[3] = 1.0;
    N[0] = 1.0;
    N[1] = 2.0;
    N[2] = 3.0;
    N[3] = 4.0;
    for (int i = 0; i < imax; ++i)
    {
        for (int k = 0; k < kmax; ++k)
        {
            printf("M_%d,%d: %0.1f\n", i, k, M[i * kmax + k]);
        }
    }
    for (int k = 0; k < kmax; ++k)
    {
        for (int j = 0; j < jmax; ++j)
        {
            printf("N_%d,%d: %0.1f\n", k, j, N[k * jmax + j]);
        }
    }
    printf("P before\n");
    for (int i = 0; i < imax; ++i)
    {
        for (int j = 0; j < jmax; ++j)
        {
            printf("P_%d,%d: %0.1f\n", i, j, P[i * jmax + j]);
        }
    }

    MatmulColthreadStub(M, N, P, imax, kmax, jmax);
    for (int i = 0; i < imax; ++i)
    {
        for (int j = 0; j < jmax; ++j)
        {
            printf("P_%d,%d: %0.1f\n", i, j, P[i * jmax + j]);
        }
    }
    return 0;
}