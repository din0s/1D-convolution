#include <algorithm>

/**
 * This method convolutes the two given float arrays.
 * The result is saved in the third array.
 */
void myConvolve(float *a, float *b, float *res, int aLen, int bLen)
{
    // Calculate the length of the result
    int abMax = std::max(aLen, bLen);
    int convLen = aLen + bLen - 1;
    for (int n = 0; n < convLen; ++n)
    {
        float prod = 0;

        // Find the minimum amount of iterations needed
        int kMax = std::min(abMax, n);
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
