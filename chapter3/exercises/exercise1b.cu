/* Write a kernel that has one thread to produce each output matrix col */

__global__ void MatmulColthreadKernel(float *M, float *N, float *P, int imax, int kmax, int jmax)
{
    /* ik, kj -> ij */
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < jmax)
    {
        for (int i = 0; i < imax; ++j)
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

    int thread_block_size = 256;
    dim3 dimsGrid(ceil(imax / thread_block_size), 1, 1);
    dim3 dimsBlock(thread_block_size, 1, 1);
    MatmulColthreadKernel<<<dimsGrid, dimsBlock>>>(M_d, N_d, P_d, imax, kmax, jmax);

    cudaMemcpy(P, P_d, psize, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}