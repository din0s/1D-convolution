clear; clf; close;

% Load audio files
[sample_audio, F_sample] = audioread("./sample_audio.wav");
[pink_noise, F_noise] = audioread("./pink_noise.wav");
[result, F_result] = audioread("./pinkNoise_sampleAudio.wav");

% Calculate convolution with built-in function
convo = conv(sample_audio, pink_noise);

figure(1); % Let's compare the results visually
subplot(4,1,1), plot(sample_audio), xlabel("sample audio");
subplot(4,1,2), plot(pink_noise), xlabel("pink noise");
subplot(4,1,3), plot(result), xlabel("result from C++");
subplot(4,1,4), plot(convo), xlabel("result from Matlab");

figure(2); % Plot a sample of each result on top of one another
result_sample = result(5000:10000); t1 = 1:length(result_sample);
convo_sample = convo(5000:10000); t2 = 1:length(convo_sample);
subplot(2,1,1), plot(t1, result_sample, 'b', t2, convo_sample, 'r.');
subplot(2,1,2), plot(convo_sample - result_sample), ylim([-0.01 0.01]);

figure(3); % Compare the frequency domains of both results
result_freq = fourier(result); convo_freq = fourier(result);
subplot(3,1,1), plot(abs(result_freq)), xlabel("freq domain from C++");
subplot(3,1,2), plot(abs(convo_freq)), xlabel("freq domain from Matlab");
subplot(3,1,3), plot(abs(result_freq) - abs(convo_freq)), ylim([-0.01 0.01]);

figure(4); % Perform convolution using multiplication in the freq. domain
pad = length(sample_audio) + length(pink_noise) - 1;
sample_freq = fft(sample_audio, pad);
noise_freq = fft(pink_noise, pad);
convo_freq = sample_freq .* noise_freq;
convo_time = ifft(convo_freq, pad);
subplot(3,1,1), plot(convo), xlabel("result from convolution");
subplot(3,1,2), plot(convo_time), xlabel("result from dft");
subplot(3,1,3), plot(convo - convo_time), xlabel("difference"), ylim([-0.01 0.01]);

function [Y] = fourier(X)
Y = fft(X);
Y = Y(1:ceil(length(Y)/2));
end
