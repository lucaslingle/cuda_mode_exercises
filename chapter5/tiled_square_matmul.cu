#define TILE_WIDTH 16

__global__ void matmulKernel(float *M, float *N, float *P, int Width)
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