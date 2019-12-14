#include "lib/Convolution.cu"
#include "Utils.h"

void convolveFile(std::string type, float *audio, float *noise, float *res, int aSize, int nSize, int cSize, AudioFile<float> &file)
{
        std::cout << "Convoluting with " << type << " noise ... ";
        myConvolve(audio, noise, res, aSize, nSize);
        std::cout << "Done!" << std::endl;
        file.samples[0] = std::vector<float>(res, res + cSize);
        file.save("./" + type + "Noise_sampleAudio.wav", AudioFileFormat::Wave);
}

int main(int argc, char const *argv[])
{
    if (Utils::selectModule() == VECTOR_CONV)
    {
        int n = Utils::readNum();
        float *a; cudaMallocManaged(&a, n * sizeof(float));
        float *b; cudaMallocManaged(&b, 5 * sizeof(float));
        float *res; cudaMallocManaged(&res, (n + 4) * sizeof(float));

        for (int i = 0; i < 5; ++i)
        {
            b[i] = 0.2;
        }

        Utils::generate(a, n);
        std::cout << "Generated a random array A:" << std::endl;
        Utils::print(a, n);

        myConvolve(a, b, res, n, 5);
        std::cout << "Convolution with [ 0.2 0.2 0.2 0.2 0.2 ] is:" << std::endl;
        Utils::print(res, n + 4);

        cudaFree(a);
        cudaFree(b);
        cudaFree(res);
    }
    else
    {
        AudioFile<float> *audioWav = Utils::readWav("./sample_audio.wav");
        AudioFile<float> *noiseWav = Utils::readWav("./pink_noise.wav");
        if (audioWav == nullptr || noiseWav == nullptr)
        {
            return 1;
        }

        AudioFile<float> resultWav;
        resultWav.setSampleRate(audioWav->getSampleRate());
        std::cout << "Audio files read successfully." << std::endl;

        std::vector<float> audioVec = audioWav->samples[0];
        std::vector<float> noiseVec = noiseWav->samples[0];
        int aSize = audioVec.size();
        int nSize = noiseVec.size();
        int cSize = aSize + nSize - 1;

        float *audio; cudaMallocManaged(&audio, aSize * sizeof(float)); Utils::copy2Arr(audioVec, audio, aSize);
        float *noise; cudaMallocManaged(&noise, nSize * sizeof(float)); Utils::copy2Arr(noiseVec, noise, nSize);
        float *res; cudaMallocManaged(&res, cSize * sizeof(float));

        convolveFile("pink", audio, noise, res, aSize, nSize, cSize, resultWav);

        std::cout << "Generating white noise ... ";
        Utils::generate(noise, nSize);
        std::cout << "Done!" << std::endl;

        convolveFile("white", audio, noise, res, aSize, nSize, cSize, resultWav);

        delete audioWav;
        delete noiseWav;
        cudaFree(audio);
        cudaFree(noise);
        cudaFree(res);
    }
    return 0;
}
