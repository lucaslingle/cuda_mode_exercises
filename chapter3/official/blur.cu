#define BLUR_SIZE 1

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h)
{
    int row = blockIdx.y * blockDim.y * threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < w && row < h)
    {
        int pixVal = 0;
        int pixels = 0;
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
        {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol)
            {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w)
                {
                    pixVal += in[curRow * w + curCol];
                    ++pixels;
                }
            }
        }
        out[row * w + col] = (unsigned char)(pixVal / pixels);
    }
};