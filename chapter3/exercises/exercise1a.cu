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

float *MatmulRowthreadStub(float *M, float *N, float *P, int imax, int kmax, int jmax)
{
    int thread_block_size = 256;
    dim3 dimsGrid(ceil(imax / thread_block_size), 1, 1);
    dim3 dimsBlock(thread_block_size, 1, 1);
    MatmulRowthreadKernel<<<dimsGrid, dimsBlock>>>(M, N, P, imax, kmax, jmax);
}