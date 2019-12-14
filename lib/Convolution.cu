__global__
void cudaConvolve(float *a, float *b, float *res, int aLen, int bLen)
{
    int abMax = max(aLen, bLen);
    int convLen = aLen + bLen - 1;
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int n = index; n < convLen; n += step)
    {
        float prod = 0;
        int kMax = min(abMax, n);
        for (int k = 0; k <= kMax; ++k)
        {
            if (k < aLen && n - k < bLen)
            {
                prod += a[k] * b[n - k];
            }
        }
        res[n] = prod;
    }
}

void myConvolve(float *a, float *b, float *res, int aLen, int bLen)
{
    int blockSize = 256;
    int numBlocks = ((aLen + bLen - 1) + blockSize - 1) / blockSize;
    cudaConvolve<<<numBlocks, blockSize>>>(a, b, res, aLen, bLen);
    cudaDeviceSynchronize();
}
