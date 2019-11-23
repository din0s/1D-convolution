#include <random>
#include "iostream"
#include "AudioFile.h"

#define DEBUG 1

using namespace std;

int read_num()
{
    int n;
    do
    {
        cout << "Please enter a number N > 10: ";
        cin >> n;
        if (cin.fail())
        {
            cin.clear();
            cin.ignore();
            n = 0;
        }
    } while (n <= 10);
    return n;
}

void print(vector<float> *arr)
{
    if (!DEBUG)
        return;

    cout << "[ ";
    for (int i = 0; i < arr->size(); i++)
    {
        cout << (*arr)[i] << ' ';
    }
    cout << "]" << endl;
}

vector<float> *generate(int n)
{
    random_device rnd;
    mt19937 gen(rnd());
    uniform_real_distribution<float> dis(-1, 1);
    vector<float> *result = new vector<float>(n);
    generate(result->begin(), result->end(), [&dis, &gen]() { return dis(gen); });
    return result;
}

AudioFile<float> *readWav(string path)
{
    AudioFile<float> *file = new AudioFile<float>();
    if (file->load(path))
    {
        return file;
    }
    cerr << "Cannot read file from path: " << path << endl;
    delete file;
    return nullptr;
}

vector<float> *myConvolve(vector<float> *a, vector<float> *b)
{
    int aLen = a->size();
    int bLen = b->size();
    int abMax = max(aLen, bLen);
    int convLen = aLen + bLen - 1;
    vector<float> *result = new vector<float>;
    for (int n = 0; n < convLen; ++n)
    {
        float prod = 0;
        int kMax = min(abMax, n);
        for (int k = 0; k <= kMax; ++k)
        {
            if (k < aLen && n - k < bLen)
            {
                prod += (*a)[k] * (*b)[n - k];
            }
        }
        result->push_back(prod);
    }
    return result;
}

int first_task()
{
    int n = read_num();
    vector<float> *a = generate(n);
    vector<float> b = vector<float>(5, 1.0 / 5);
    vector<float> *result = myConvolve(a, &b);

    cout << "Generated a random array A:" << endl;
    print(a);

    cout << "Convolution with [ 0.2 0.2 0.2 0.2 0.2 ] is:" << endl;
    print(result);

    delete a;
    delete result;
    return 0;
}

int second_task()
{
    AudioFile<float> *audio = readWav("./sample_audio2.wav");
    AudioFile<float> *noise = readWav("./pink_noise.wav");
    if (audio == nullptr || noise == nullptr)
    {
        cout << "Program exiting abnormally." << endl;
        return 1;
    }
    cout << "Audio files read successfully." << endl;

    AudioFile<float> result;
    result.setSampleRate(audio->getSampleRate());

    cout << "Convoluting with pink noise." << endl;
    result.samples[0] = *myConvolve(&audio->samples[0], &noise->samples[0]);
    result.save("pinkNoise_sampleAudio.wav", AudioFileFormat::Wave);
    cout << "Convolution with pink noise completed." << endl;

    cout << "Generating white noise." << endl;
    vector<float> *white = generate(noise->samples[0].size());

    cout << "Convoluting with white noise." << endl;
    result.samples[0] = *myConvolve(&audio->samples[0], white);
    result.save("whiteNoise_sampleAudio.wav", AudioFileFormat::Wave);
    cout << "Convolution with white noise completed." << endl;

    delete audio;
    delete noise;
    delete white;
    return 0;
}

int main(int argc, char const *argv[])
{
    // return first_task();
    return second_task();
}
