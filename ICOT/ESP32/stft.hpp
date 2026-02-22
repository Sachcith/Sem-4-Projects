#ifndef stft_H
#define stft_H
#include "cmath"
#include "fftw3.h"
#include "Matrix.hpp"

// STFT Class
template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE>
class stft{
    private:
        // Constants
        static constexpr std::size_t NUM_FRAMES = 1 + (TOTAL_SAMPLES - FFT_SIZE) / HOP_SIZE;
        static constexpr std::size_t NUM_BINS = FFT_SIZE/2 + 1;
        Matrix<T,TOTAL_SAMPLES,1> signal;

        // Private Methods
        Matrix<T,FFT_SIZE,1> window_with_hann(std::size_t start,Matrix<T,FFT_SIZE,1>hann);
        Matrix<T,FFT_SIZE,1> hann_generator();
    public:
        // Public Methods
        void setSignal(Matrix<T,TOTAL_SAMPLES,1>input);
        
        // Compute STFT Function
        Matrix<T,NUM_FRAMES,NUM_BINS> compute(){
            Matrix<T,NUM_FRAMES,NUM_BINS> output;

            float in[FFT_SIZE];
            fftwf_complex out[NUM_BINS];

            fftwf_plan plan = fftwf_plan_dft_r2c_1d(FFT_SIZE,in,out,FFTW_ESTIMATE);

            Matrix<T,FFT_SIZE,1> hann = hann_generator();

            for(size_t i=0;i<NUM_FRAMES;i++){
                size_t start = i * HOP_SIZE;
                
                Matrix<T,FFT_SIZE,1> window = window_with_hann(start,hann);

                for(size_t j=0;j<FFT_SIZE;j++){
                    in[j] = static_cast<float>(window[j][0]);
                }

                fftwf_execute(plan);

                for(size_t j=0;j<NUM_BINS;j++){
                    output[i][j] = std::sqrt(out[j][0]*out[j][0] + out[j][1]*out[j][1]);
                }

            }

            fftwf_destroy_plan(plan);

            return output;
        }
};

// Set Signal Function
template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE>
void stft<T,TOTAL_SAMPLES,FFT_SIZE,HOP_SIZE>::setSignal(Matrix<T,TOTAL_SAMPLES,1> input){
    for(size_t i=0;i<TOTAL_SAMPLES;i++){
        signal[i][0] = input[i][0];
    }
}

// Compute Window Function
template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE>
Matrix<T,FFT_SIZE,1> stft<T,TOTAL_SAMPLES,FFT_SIZE,HOP_SIZE>::window_with_hann(std::size_t start,Matrix<T,FFT_SIZE,1>hann){
    Matrix<T,FFT_SIZE,1> output;
    for(size_t i=0;i<FFT_SIZE;i++){
        output[i][0] = signal[start+i][0] * hann[i][0];
    }
    return output;
}

// Hann Generator Function
// Used to smooth the frequency domain
template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE>
Matrix<T,FFT_SIZE,1> stft<T,TOTAL_SAMPLES,FFT_SIZE,HOP_SIZE>::hann_generator(){
    Matrix<T,FFT_SIZE,1> output;
    T pi = static_cast<T>(3.14159265358979323846);
    for(std::size_t i=0;i<FFT_SIZE;i++){
        //output[i][0] = 1; // To Disable Hann uncomment this line. 
        output[i][0] = static_cast<T>(0.5) - static_cast<T>(0.5) * std::cos((static_cast<T>(2)*pi*i)/(FFT_SIZE-1));
    }
    return output;
}

#endif