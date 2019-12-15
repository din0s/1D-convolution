#include "lib/Convolution.cu"
#include "Utils.h"

/**
 * This method is responsible for convoluting a float array of an audio file with an array of noise and then
 * saving its result to a new WAV file. This is done to increase re-usability of the code since this task
 * needs to be done twice, once for the provided pink noise and noce for the generated white noise.
 */
void convolveFile(std::string type, float *audio, float *noise, float *res, int aSize, int nSize, int cSize, AudioFile<float> &file)
{
    std::cout << "Convoluting with " << type << " noise ... ";
    myConvolve(audio, noise, res, aSize, nSize);                             // Call myConvolve from Convolution.cu
    file.samples[0] = std::vector<float>(res, res + cSize);                  // Create a vector from the float[] result
    file.save("./" + type + "Noise_sampleAudio.wav", AudioFileFormat::Wave); // Save result to file
    std::cout << "Done!" << std::endl;
}

int main(int argc, char const *argv[])
{
    // Prompt the user to select which module to execute
    if (Utils::selectModule() == VECTOR_CONV) // FIRST TASK
    {
        // Read the size of the 1st array
        int n = Utils::readNum();

        // Allocate memory for all arrays
        float *a; cudaMallocManaged(&a, n * sizeof(float));
        float *b; cudaMallocManaged(&b, 5 * sizeof(float));
        float *res; cudaMallocManaged(&res, (n + 4) * sizeof(float));

        // Populate the B array
        for (int i = 0; i < 5; ++i)
        {
            b[i] = 0.2;
        }

        Utils::generate(a, n); // Generate a random array of float numbers
        std::cout << "Generated a random array A:" << std::endl;
        Utils::print(a, n);

        myConvolve(a, b, res, n, 5); // Call myConvolve from Convolution.cu
        std::cout << "Convolution with [ 0.2 0.2 0.2 0.2 0.2 ] is:" << std::endl;
        Utils::print(res, n + 4); // Display the result to the user

        // Free memory
        cudaFree(a);
        cudaFree(b);
        cudaFree(res);
    }
    else // SECOND TASK
    {
        // Attempt to read the WAV files, if present
        AudioFile<float> *audioWav = Utils::readWav("./sample_audio.wav");
        AudioFile<float> *noiseWav = Utils::readWav("./pink_noise.wav");
        if (audioWav == nullptr || noiseWav == nullptr)
        {
            // Audio files not present, exit with error
            return 1;
        }

        AudioFile<float> resultWav;
        resultWav.setSampleRate(audioWav->getSampleRate()); // Set the result's sample rate based on the song's rate
        std::cout << "Audio files read successfully." << std::endl;

        // Extract the samples from both files
        std::vector<float> audioVec = audioWav->samples[0];
        std::vector<float> noiseVec = noiseWav->samples[0];
        int aSize = audioVec.size();
        int nSize = noiseVec.size();
        int cSize = aSize + nSize - 1;

        // Convert vectors to float arrays and allocate memory for the result
        float *audio; cudaMallocManaged(&audio, aSize * sizeof(float)); Utils::copy2Arr(audioVec, audio, aSize);
        float *noise; cudaMallocManaged(&noise, nSize * sizeof(float)); Utils::copy2Arr(noiseVec, noise, nSize);
        float *res; cudaMallocManaged(&res, cSize * sizeof(float));

        convolveFile("pink", audio, noise, res, aSize, nSize, cSize, resultWav);

        std::cout << "Generating white noise ... ";
        Utils::generate(noise, nSize); // Generate the white noise array with the same length as the pink noise
        std::cout << "Done!" << std::endl;

        convolveFile("white", audio, noise, res, aSize, nSize, cSize, resultWav);

        // Free memory
        delete audioWav;
        delete noiseWav;
        cudaFree(audio);
        cudaFree(noise);
        cudaFree(res);
    }
    return 0;
}
