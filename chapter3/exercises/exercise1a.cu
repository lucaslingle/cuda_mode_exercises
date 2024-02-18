/* Write a kernel that has one thread to produce each output matrix row */

__global__ void MatmulRowthreadKernel(float *M, float *N, float *P, int imax, int kmax, int jmax)
{
    /* ik, kj -> ij */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < imax)
    {
        for (int j = 0; j < jmax; ++j)
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

void MatmulRowthreadStub(float *M, float *N, float *P, int imax, int kmax, int jmax)
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
    MatmulRowthreadKernel<<<dimsGrid, dimsBlock>>>(M_d, N_d, P_d, imax, kmax, jmax);

    cudaMemcpy(P, P_d, psize, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);
}