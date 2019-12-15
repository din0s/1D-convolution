__global__
/**
 * This is the kernel that convolutes the two given float arrays.
 * The result is saved in the third array.
 */
void cudaConvolve(float *a, float *b, float *res, int aLen, int bLen)
{
    // Calculate the length of the result
    int abMax = max(aLen, bLen);
    int convLen = aLen + bLen - 1;

    // Find the starting point and the step of the loop
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (int n = index; n < convLen; n += step)
    {
        float prod = 0;

        // Find the minimum amount of iterations needed
        int kMax = min(abMax, n);
        for (int k = 0; k <= kMax; ++k)
        {
            // Make sure we're in bounds for both arrays,
            // otherwise there's no overlap between the two.
            if (k < aLen && n - k < bLen)
            {
                prod += a[k] * b[n - k];
            }
        }
        res[n] = prod;
    }
}

/**
 * This method calls the CUDA kernel for the convolution, after
 * calculating the proper amount of blocks and threads needed.
 */
void myConvolve(float *a, float *b, float *res, int aLen, int bLen)
{
    int blockSize = 256;
    int numBlocks = ((aLen + bLen - 1) + blockSize - 1) / blockSize;
    cudaConvolve<<<numBlocks, blockSize>>>(a, b, res, aLen, bLen);
    cudaDeviceSynchronize(); // Wait for all CUDA cores to finish
}
