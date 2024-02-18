#include <stdio.h>

__global__ void MatvecKernel(float *A, float *B, float *C, unsigned int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        float acc = 0;
        for (int j = 0; j < N; ++j)
        {
            acc += B[i * N + j] * C[j];
        }
        A[i] = acc;
    }
}

void MatvecStub(float *A, float *B, float *C, unsigned int N)
{
    float *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, N * sizeof(float));
    cudaMalloc((void **)&B_d, N * N * sizeof(float));
    cudaMalloc((void **)&C_d, N * sizeof(float));

    cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, N * sizeof(float), cudaMemcpyHostToDevice);

    int block_sz = 256;
    dim3 dimGrid((N + block_sz - 1) / block_sz, 1, 1);
    dim3 dimBlock(block_sz, 1, 1);
    MatvecKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, N);

    cudaMemcpy(A, A_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
    float *A, *B, *C;
    int N = 3;

    A = (float *)malloc(N * sizeof(float));
    B = (float *)malloc(N * N * sizeof(float));
    C = (float *)malloc(N * sizeof(float));

    B[0] = 3.0;
    B[1] = 0.0;
    B[2] = 0.0;
    B[3] = 0.0;
    B[4] = 5.0;
    B[5] = 0.0;
    B[6] = 0.0;
    B[7] = 0.0;
    B[8] = 7.0;

    C[0] = 1.0;
    C[1] = 2.0;
    C[2] = 3.0;

    for (int i = 0; i < N * N; ++i)
    {
        printf("B_%d,%d: %0.1f\n", i / N, i % N, B[i]);
    }
    for (int i = 0; i < N; ++i)
    {
        printf("C_%d: %0.1f\n", i, C[i]);
    }
    printf("A before\n");
    for (int i = 0; i < N; ++i)
    {
        printf("A_%d: %0.1f\n", i, A[i]);
    }

    MatvecStub(A, B, C, N);
    for (int i = 0; i < N; ++i)
    {
        printf("A_%d: %0.1f\n", i, A[i]);
    }
    return 0;
}