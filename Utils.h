#include "lib/AudioFile.h"
#include <iostream>
#include <time.h>

#define VECTOR_CONV 1
#define PRINT_SIZE 1000
#define CUTOFF_SIZE 100

class Utils
{
private:
    static int input()
    {
        int n;
        std::cin >> n;
        if (std::cin.fail())
        {
            std::cin.clear();
            std::cin.ignore();
            n = 0;
        }
        return n;
    }

public:
    static void copy2Arr(std::vector<float> &vec, float *arr, int len)
    {
        for (int i = 0; i < len; ++i)
        {
            arr[i] = vec[i];
        }
    }

    static void generate(float *a, int n)
    {
        srand(time(NULL));
        for (int i = 0; i < n; ++i)
        {
            double sign = (rand() % 2) == 0 ? 1.0 : -1.0;
            double divisor = sign * RAND_MAX;
            a[i] = rand() / divisor;
        }
    }

    static void print(float *arr, int size)
    {
        std::cout << "[ ";
        for (int i = 0; i < size; i++)
        {
            std::cout << arr[i] << ' ';

            if (size > PRINT_SIZE && i == CUTOFF_SIZE - 1)
            {
                std::cout << "... ";
                i = size - CUTOFF_SIZE;
            }
        }
        std::cout << "]" << std::endl;
    }

    static int readNum(int lowBound)
    {
        int n;
        do
        {
            std::cout << "Please enter a number N > " << lowBound << ": ";
            n = input();
        } while (n <= lowBound);
        return n;
    }

    static int readNum()
    {
        return readNum(10);
    }

    static AudioFile<float> *readWav(std::string path)
    {
        AudioFile<float> *file = new AudioFile<float>();
        if (file->load(path))
        {
            return file;
        }
        delete file;
        return nullptr;
    }

    static int selectModule()
    {
        int n;
        do
        {
            std::cout << "Options:" << std::endl
                      << "1) Vector convolution A[N] with B = [ 0.2 0.2 0.2 0.2 0.2 ]" << std::endl
                      << "2) WAV convolution with audio file and noise" << std::endl
                      << "Type '1' or '2':" << std::endl;
            n = input();
        } while (n != 1 && n != 2);
        return n;
    }
};
