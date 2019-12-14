#include <algorithm>

void myConvolve(float *a, float *b, float *res, int aLen, int bLen)
{
    int abMax = std::max(aLen, bLen);
    int convLen = aLen + bLen - 1;
    for (int n = 0; n < convLen; ++n)
    {
        float prod = 0;
        int kMax = std::min(abMax, n);
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
