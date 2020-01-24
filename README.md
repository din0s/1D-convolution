# 1D-convolution
This project is an implementation of an one-dimensional convolution in C++ and CUDA.

Signals & Systems, 5th semester of Computer Science Dept. @ Aristotle University of Thessaloniki

## Implementation requirements
The first task requires the generation of a random float array which is then convoluted with: `[0.2 0.2 0.2 0.2 0.2]`

In order to accelerate the execution time of the algorithm, we need to reduce the amount of elements checked in each iteration.
This can be done by limiting the range of products calculated to be within the bounds where the vectors overlap.

Once the convolution method is implemented, we can use it in order to convolve two WAV files instead of random numbers.
The repository [adamstark/AudioFile](https://github.com/adamstark/AudioFile) was used in order to load the files into memory
as float vectors, which can then be passed as arguments to the convolution method.

The final step to this was to port the C++ implementation to CUDA, which was made easy by following
[this guide](https://devblogs.nvidia.com/even-easier-introduction-cuda/) from NVIDIA's developer blog.
Both implementations mostly relied on the same codebase (written in C++) so as to improve maintainability.
The main differences are in the convolution method itself, as well as the memory allocation methods.

## Result correctness
To verify that everything worked as intended, the results were cross-checked with MATLAB's built-in convolution function.
Here's a sample plot highlighting the difference to the output, which is assumed to be due to floating point errors:

![diff](https://i.imgur.com/UuUMVch.png)

More insight on the verification process can be seen in the m-file `analysis.m`.

## C++ / CUDA execution time
A test was conducted with the following specs:
- CPU: Intel Core i7-770K @ 4.8 GHz
- RAM: 16GB DDR4 @ 3000MHz (CAS Latency: 15)
- GPU: NVIDIA GTX 1080 @ Core 2000 MHz - Memory 5600 MHz

The two WAV files that were convoluted required about `13.3 minutes` in C++ and just `1.3 seconds` in CUDA.
