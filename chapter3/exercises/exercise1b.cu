/* Write a kernel that has one thread to produce each output matrix col */

__global__ void MatmulRowthreadKernel(float *M, float *N, float *P, int imax, int kmax, int jmax)
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

float *MatmulColthreadStub(float *M, float *N, float *P, int imax, int kmax, int jmax)
{
    int thread_block_size = 256;
    dim3 dimsGrid(ceil(imax / thread_block_size), 1, 1);
    dim3 dimsBlock(thread_block_size, 1, 1);
    MatmulRowthreadKernel<<<dimsGrid, dimsBlock>>>(M, N, P, imax, kmax, jmax);
}