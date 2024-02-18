__global__ void colortoGreyscaleConversion(unsigned char *Pout, unsigned char *Pin, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width)
    {
        int greyOffset = row * width + col;
        int rgbOffset = greyOffset * 3;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        Pout[greyOffset] = 0.21f * r + 0.71f * g + 0.07 * b;
    }
};
