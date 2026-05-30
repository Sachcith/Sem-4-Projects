#ifndef stft_H
#define stft_H
#include "cmath"
#include "esp_dsp.h"
#include "Matrix.hpp"
#include "feature.hpp"
#include "esp_task_wdt.h"
#define YIELD() delay(1)
// #define YIELD() vTaskDelay()

// STFT Class for ESP32 using ESP-DSP
template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE, std::size_t FINAL_SIZE>
class stft {
private:
    static constexpr std::size_t NUM_FRAMES = 1 + TOTAL_SAMPLES / HOP_SIZE;
    static constexpr std::size_t NUM_BINS = FFT_SIZE / 2 + 1;
    static constexpr std::size_t FINAL = FINAL_SIZE;
    static constexpr std::size_t PAD_LEN = FFT_SIZE / 2;

    Matrix<T, NUM_BINS, 1> buffer1;
    Matrix<T, NUM_BINS, 1> buffer2;
    Matrix<T, NUM_BINS, 1> buffer3;
    Matrix<T, NUM_BINS, 1> buffer4;
    Matrix<T, NUM_BINS, 1> buffer5;
    Matrix<T, NUM_BINS, 1> buffer6;
    Matrix<T, NUM_BINS, 1> buffer7;
    Matrix<T, NUM_BINS, 1> buffer8;
    Matrix<T, NUM_BINS, 1> buffer9;

    void window_with_hann(int start, const Matrix<T, FFT_SIZE, 1> &hann, const Matrix<T, FINAL, 1> &signal, Matrix<T, FFT_SIZE, 1> &output);
    Matrix<T, FFT_SIZE, 1> hann_generator();

public:
    void compute(const Matrix<T, FINAL, 1> &signal, Matrix<T, NUM_FRAMES * NUM_BINS, 1> &output) {
        output.reshape(NUM_FRAMES, NUM_BINS);

        float in[FFT_SIZE];
        float out_real[FFT_SIZE];
        float out_imag[FFT_SIZE];

        // Init FFT (forward real-to-complex)
        dsps_fft2r_init_fc32(NULL, FFT_SIZE);

        Matrix<T, FFT_SIZE, 1> hann = hann_generator();
        Matrix<T, FFT_SIZE, 1> window;

        for (std::size_t i = 0; i < NUM_FRAMES; i++) {
            int start = static_cast<int>(i) * static_cast<int>(HOP_SIZE) - static_cast<int>(PAD_LEN);
            window_with_hann(start, hann, signal, window);

            for (std::size_t j = 0; j < FFT_SIZE; j++) {
                in[j] = static_cast<float>(window[j][0]);
            }

            // Perform FFT using ESP-DSP
            float fft_buffer[FFT_SIZE * 2]; // real+imag interleaved
            for (std::size_t j = 0; j < FFT_SIZE; j++) {
                fft_buffer[2 * j] = in[j];  // real
                fft_buffer[2 * j + 1] = 0;  // imag
            }

            
            dsps_fft2r_fc32(fft_buffer, FFT_SIZE);
            dsps_bit_rev_fc32(fft_buffer, FFT_SIZE);
            dsps_cplx2reC_fc32(fft_buffer, FFT_SIZE);

            // Copy magnitude to output
            for (std::size_t j = 0; j < NUM_BINS; j++) {
                float real = fft_buffer[2 * j];
                float imag = fft_buffer[2 * j + 1];
                output[i][j] = std::sqrt(real * real + imag * imag);
            }
        }
    }
};

// Compute Window Function
template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE, std::size_t FINAL_SIZE>
void stft<T, TOTAL_SAMPLES, FFT_SIZE, HOP_SIZE, FINAL_SIZE>::window_with_hann(int start, const Matrix<T, FFT_SIZE, 1> &hann, const Matrix<T, FINAL, 1> &signal, Matrix<T, FFT_SIZE, 1> &output) {
    for (std::size_t i = 0; i < FFT_SIZE; i++) {
        int index = start + i;
        // while (index < 0 || index >= TOTAL_SAMPLES) {
        //     if (index < 0) index = -index;
        //     else index = 2 * TOTAL_SAMPLES - index - 2;
        // }
        // output[i][0] = signal[index][0] * hann[i][0];
        
        // New (zero padding like librosa)
        if(index < 0 || index >= TOTAL_SAMPLES){
            output[i][0] = 0.0f;
        } else {
            output[i][0] = signal[index][0] * hann[i][0];
        }
    }
}

// Hann Generator
template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE, std::size_t FINAL_SIZE>
Matrix<T, FFT_SIZE, 1> stft<T, TOTAL_SAMPLES, FFT_SIZE, HOP_SIZE, FINAL_SIZE>::hann_generator() {
    Matrix<T, FFT_SIZE, 1> output;
    T pi = static_cast<T>(3.14159265358979323846);
    for (std::size_t i = 0; i < FFT_SIZE; i++) {
        output[i][0] = 0.5 - 0.5 * std::cos((2 * pi * i) / (FFT_SIZE-1));
    }
    return output;
}

// Zoom STFT (same as your previous implementation)
template <class T, std::size_t N, std::size_t Input_row, std::size_t Input_col, std::size_t Output_row, std::size_t Output_col>
class zoom_stft {
public:
    Matrix<T, Output_row * Output_col, 1> output;

    Matrix<T, Output_row * Output_col, 1> zoom(Matrix<T, N, 1> &input) {
        input.reshape(Input_row, Input_col);
        output.reshape(Output_row, Output_col);
        T row_scale = static_cast<T>(Input_row - 1) / (Output_row - 1);
        T col_scale = static_cast<T>(Input_col - 1) / (Output_col - 1);

        for (size_t i = 0; i < Output_row; i++) {
            for (size_t j = 0; j < Output_col; j++) {
                T r = i * row_scale;
                size_t r0 = static_cast<size_t>(std::floor(r));
                T wr = r - r0;
                size_t r1 = (r0 + 1 >= Input_row) ? r0 : r0 + 1;

                T c = j * col_scale;
                size_t c0 = static_cast<size_t>(std::floor(c));
                T wc = c - c0;
                size_t c1 = (c0 + 1 >= Input_col) ? c0 : c0 + 1;

                output[i][j] = (1 - wr) * (1 - wc) * input[r0][c0] +
                               (1 - wr) * wc * input[r0][c1] +
                               wr * (1 - wc) * input[r1][c0] +
                               wr * wc * input[r1][c1];
            }
        }
        input.reset_shape();
        output.reset_shape();
        return output;
    }
};


// Convert to log thingy
template <class T, std::size_t N>
void log_thingy(Matrix<T,N,1> &input){
    T max_thingy = max_value(input);
    T zero_error = 1e-10;
    max_thingy = max_thingy + zero_error;
    YIELD(); for(std::size_t i=0;i<N;i++){
        T val = input[i][0] + zero_error;
        input[i][0] = 20*std::log10(val/max_thingy);
    }
}

// Centroid Calculator thingy
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class centroid_thingy{
    private:
        Matrix<T,NUM_FRAMES,1> output;
    public:
        T compute(Matrix<T,N,1> &input,std::size_t sample_rate){
            input.reshape(NUM_FRAMES,NUM_BINS);
            std::size_t FFT_SIZE = (NUM_BINS-1)*2;
            YIELD(); for(std::size_t i=0;i<NUM_FRAMES;i++){
                T mag_total = 0;
                YIELD(); for(std::size_t j=0;j<NUM_BINS;j++){
                    T fi = (sample_rate*j)/(FFT_SIZE);
                    output[i][0] = output[i][0] + input[i][j]*fi;
                    mag_total = mag_total + input[i][j];
                }
                if(mag_total==0) output[i][0] = 0;
                else output[i][0] = output[i][0]/mag_total;
            }

            input.reset_shape();
            return mean(output);
        }
};

// Band Energy Ratio (BER)
// Band 1 = 150Hz to 300Hz
// Band 2 = 400Hz to 500Hz
// Freq thingy = j*sample_rate/FFT
// Freq thingy = 16000/1024 = 15.625
// Band 1 = index 10 = Freq thingy * 10 to index 22 = Freq thingy * 22
// Band 2 = index 26 = Freq thingy * 26 to index 32 = Freq thingy * 32
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class BER{
    private:
        Matrix<T,NUM_FRAMES,1> ber;
    public:
        T compute(Matrix<T,N,1> &input, std::size_t sample_rate){
            input.reshape(NUM_FRAMES,NUM_BINS);
            YIELD(); for(std::size_t i=0;i<NUM_FRAMES;i++){
                T band_1 = 0;
                YIELD(); for(std::size_t j=10;j<=22;j++){
                    band_1 = band_1 + input[i][j]*input[i][j];
                }
                T band_2 = 0;
                YIELD(); for(std::size_t j=26;j<=32;j++){
                    band_2 = band_2 + input[i][j]*input[i][j];
                }
                if(band_1 == 0) ber[i][0] = 0;
                else ber[i][0] = band_2/band_1;
            }
            input.reset_shape();
            return mean(ber);
        }
};


#endif