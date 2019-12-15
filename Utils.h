#include "lib/AudioFile.h"
#include <iostream>
#include <time.h>

#define VECTOR_CONV 1   // Constant, 1 for vector convolution, 2 for WAV files
#define PRINT_SIZE 1000 // Arrays up to PRINT_SIZE won't be trimmed when displaying
#define CUTOFF_SIZE 100 // Only display elements [0 : CUTOFF_SIZE], [SIZE - CUTOFF_SIZE : SIZE] for trimmed arrays

class Utils
{
private:
    /**
     * This method reads an integer from the user.
     * If an invalid value is entered, this returns a default value of 0.
     */
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
    /**
     * This method copies the contents of a vector to an array.
     * This is needed when populating an array that's shared between the host and the device in CUDA.
     */
    static void copy2Arr(std::vector<float> &vec, float *arr, int len)
    {
        for (int i = 0; i < len; ++i)
        {
            arr[i] = vec[i];
        }
    }

    /**
     * This method generates an array of pseudorandom floating point numbers.
     */
    static void generate(float *a, int n)
    {
        srand(time(NULL)); // Seed the random number generator
        for (int i = 0; i < n; ++i)
        {
            float sign = (rand() % 2) == 0 ? 1.0 : -1.0; // Get a sign (psudo)randomly
            float divisor = sign * RAND_MAX;             // Assign a sign to the divisor
            a[i] = rand() / divisor;                     // Save the result to the array
        }
    }

    /**
     * This method prints the contents of a float array.
     */
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

    /**
     * This method reads a number from the user, given a lower bound.
     * As a result, this will repeat until the user selects a number N > lowBound.
     */
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

    /**
     * This method reads a number from the user, greater than 10. See readNum(int).
     */
    static int readNum()
    {
        return readNum(10);
    }

    /**
     * This method reads a WAV file from the given path.
     * If the file cannot be found/read, this returns a value of nullptr.
     */
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

    /**
     * This method prompts the user to select one of the two available modules/tasks.
     */
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
