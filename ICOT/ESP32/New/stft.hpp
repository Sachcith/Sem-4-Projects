// #ifndef stft_H
// #define stft_H

// #include <Arduino.h>
// #include <cmath>
// #include <esp_dsp.h>
// #include "Matrix.hpp"
// #include "feature.hpp"

// // STFT Class
// template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE, std::size_t FINAL_SIZE>
// class stft{
// private:

//     static constexpr std::size_t NUM_FRAMES = 1 + (TOTAL_SAMPLES - FFT_SIZE) / HOP_SIZE;
//     static constexpr std::size_t NUM_BINS = FFT_SIZE/2 + 1;
//     static constexpr std::size_t FINAL = max(max((NUM_FRAMES+1)*NUM_BINS,TOTAL_SAMPLES),FINAL_SIZE);

//     Matrix<T,NUM_BINS,1> buffer1;
//     Matrix<T,NUM_BINS,1> buffer2;
//     Matrix<T,NUM_BINS,1> buffer3;

//     Matrix<T,FFT_SIZE,1> window_with_hann(std::size_t start,
//                                           Matrix<T,FFT_SIZE,1> hann,
//                                           Matrix<T,FINAL,1> &signal);

//     Matrix<T,FFT_SIZE,1> hann_generator();

// public:

//     void compute(Matrix<T,FINAL,1> &signal){

//         static float fft_buffer[FFT_SIZE * 2];   // interleaved real+imag
//         Matrix<T,FFT_SIZE,1> hann = hann_generator();

//         // Initialize FFT (only first time)
//         static bool fft_init = false;
//         if(!fft_init){
//             dsps_fft2r_init_fc32(NULL, FFT_SIZE);
//             fft_init = true;
//         }

//         for(std::size_t i=0;i<NUM_FRAMES;i++){

//             std::size_t start = i * HOP_SIZE;
//             Matrix<T,FFT_SIZE,1> window = window_with_hann(start,hann,signal);

//             // Prepare complex buffer
//             for(std::size_t j=0;j<FFT_SIZE;j++){
//                 fft_buffer[2*j]   = static_cast<float>(window[j][0]);
//                 fft_buffer[2*j+1] = 0.0f;
//             }

//             // Perform FFT
//             dsps_fft2r_fc32(fft_buffer, FFT_SIZE);
//             dsps_bit_rev_fc32(fft_buffer, FFT_SIZE);
//             dsps_cplx2reC_fc32(fft_buffer, FFT_SIZE);

//             for(std::size_t j=0;j<NUM_BINS;j++){

//                 float real = fft_buffer[2*j];
//                 float imag = fft_buffer[2*j+1];

//                 if(i>=3){
//                     signal.reshape(NUM_FRAMES+1,NUM_BINS);
//                     signal[i-3][j] = buffer3[j][0];
//                     signal.reset_shape();
//                 }

//                 buffer3[j][0] = buffer2[j][0];
//                 buffer2[j][0] = buffer1[j][0];
//                 buffer1[j][0] = std::sqrt(real*real + imag*imag);
//             }
//         }

//         signal.reshape(NUM_FRAMES+1,NUM_BINS);

//         for(std::size_t i=NUM_FRAMES-3;i<NUM_FRAMES;i++){
//             for(std::size_t j=0;j<NUM_BINS;j++){
//                 signal[i][j] = buffer3[j][0];
//                 buffer3[j][0] = buffer2[j][0];
//                 buffer2[j][0] = buffer1[j][0];
//             }
//         }

//         for(std::size_t i=NUM_BINS;i<signal.rows();i++){
//             for(std::size_t j=0;j<NUM_BINS;j++){
//                 signal[i][j] = 0;
//             }
//         }

//         signal.reset_shape();
//     }
// };

// // Windowing
// template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE, std::size_t FINAL_SIZE>
// Matrix<T,FFT_SIZE,1>
// stft<T,TOTAL_SAMPLES,FFT_SIZE,HOP_SIZE,FINAL_SIZE>::window_with_hann(
//     std::size_t start,
//     Matrix<T,FFT_SIZE,1> hann,
//     Matrix<T,FINAL,1> &signal){

//     Matrix<T,FFT_SIZE,1> output;

//     for(std::size_t i=0;i<FFT_SIZE;i++){
//         output[i][0] = signal[start+i][0] * hann[i][0];
//     }
//     return output;
// }

// // Hann Generator
// template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE, std::size_t FINAL_SIZE>
// Matrix<T,FFT_SIZE,1>
// stft<T,TOTAL_SAMPLES,FFT_SIZE,HOP_SIZE,FINAL_SIZE>::hann_generator(){

//     Matrix<T,FFT_SIZE,1> output;
//     T pi = static_cast<T>(3.14159265358979323846);

//     for(std::size_t i=0;i<FFT_SIZE;i++){
//         output[i][0] =
//             static_cast<T>(0.5) -
//             static_cast<T>(0.5) *
//             std::cos((static_cast<T>(2)*pi*i)/(FFT_SIZE-1));
//     }
//     return output;
// }

// // Convert to log thingy
// template <class T, std::size_t N>
// void log_thingy(Matrix<T,N,1> &input){
//     T max_thingy = max_value(input);
//     T zero_error = 1e-5;
//     max_thingy = max_thingy + zero_error;
//     for(std::size_t i=0;i<N;i++){
//         if(input[i][0]>0){
//             T val = input[i][0] + zero_error;
//             input[i][0] = 20*std::log10(val/max_thingy);
//         }
//     }
// }


// // Centroid Calculator thingy
// template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
// class centroid_thingy{
//     private:
//         Matrix<T,NUM_FRAMES,1> output;
//     public:
//         T compute(Matrix<T,N,1> &input,std::size_t sample_rate){
//             input.reshape(NUM_FRAMES,NUM_BINS);
//             std::size_t FFT_SIZE = (NUM_BINS-1)*2;
//             for(std::size_t i=0;i<NUM_FRAMES;i++){
//                 T mag_total = 0;
//                 for(std::size_t j=0;j<NUM_BINS;j++){
//                     T fi = (sample_rate*j)/(FFT_SIZE);
//                     output[i][0] = output[i][0] + input[i][j]*fi;
//                     mag_total = mag_total + input[i][j];
//                 }
//                 if(mag_total==0) output[i][0] = 0;
//                 else output[i][0] = output[i][0]/mag_total;
//             }

//             input.reset_shape();
//             return mean(output);
//         }
// };

// // Band Energy Ratio (BER)
// // Band 1 = 150Hz to 300Hz
// // Band 2 = 400Hz to 500Hz
// // Freq thingy = j*sample_rate/FFT
// // Freq thingy = 16000/1024 = 15.625
// // Band 1 = index 10 = Freq thingy * 10 to index 22 = Freq thingy * 22
// // Band 2 = index 26 = Freq thingy * 26 to index 32 = Freq thingy * 32
// template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
// class BER{
//     private:
//         Matrix<T,NUM_FRAMES,1> ber;
//     public:
//         T compute(Matrix<T,N,1> &input, std::size_t sample_rate){
//             input.reshape(NUM_FRAMES,NUM_BINS);
//             for(std::size_t i=0;i<NUM_FRAMES;i++){
//                 T band_1 = 0;
//                 for(std::size_t j=10;j<=22;j++){
//                     band_1 = band_1 + input[i][j]*input[i][j];
//                 }
//                 T band_2 = 0;
//                 for(std::size_t j=26;j<=32;j++){
//                     band_2 = band_2 + input[i][j]*input[i][j];
//                 }
//                 if(band_1 == 0) ber[i][0] = 0;
//                 else ber[i][0] = band_2/band_1;
//             }
//             input.reset_shape();
//             return mean(ber);
//         }
// };

// // Zoom STFT Class
// template <class T, std::size_t N, std::size_t Input_row, std::size_t Input_col, std::size_t Output_row, std::size_t Output_col>
// class zoom_stft{
//     public:
//         Matrix<T,Output_row*Output_col,1> output;
//         // Zoom Function
//         Matrix<T,Output_row*Output_col,1> zoom(Matrix<T,N,1> &input){
//             input.reshape(Input_row,Input_col);
//             output.reshape(Output_row,Output_col);
//             T row_scale = static_cast<T>(Input_row-1) / (Output_row-1);
//             T col_scale = static_cast<T>(Input_col-1) / (Output_col-1);
//             for(std::size_t i=0;i<Output_row;i++){
//                 for(std::size_t j=0;j<Output_col;j++){
//                     // std::size_t r0 = i*row_scale;
//                     // T wr = r0 - std::floor(r0);
//                     // r0 = std::floor(r0); // Floor Row
//                     // std::size_t r1 = r0 + 1;// Ceil Row
//                     // if(r1 >= Input_row) r1 = Input_row-1;

//                     // std::size_t c0 = j*col_scale;
//                     // T wc = c0 - std::floor(c0);
//                     // c0 = std::floor(c0);// Floor Col
//                     // std::size_t c1 = c0 + 1;// Ceil Col
//                     // if(c1 >= Input_col) c1 = Input_col-1;
//                     T r = i * row_scale;
//                     std::size_t r0 = static_cast<std::size_t>(std::floor(r));
//                     T wr = r - r0;
//                     std::size_t r1 = r0 + 1;
//                     if (r1 >= Input_row) r1 = Input_row - 1;

//                     T c = j * col_scale;
//                     std::size_t c0 = static_cast<std::size_t>(std::floor(c));
//                     T wc = c - c0;
//                     std::size_t c1 = c0 + 1;
//                     if (c1 >= Input_col) c1 = Input_col - 1;

//                     output[i][j] = (1-wr)*(1-wc)*input[r0][c0] + (1-wr)*wc*input[r0][c1] + wr*(1-wc)*input[r1][c0] + wr*wc*input[r1][c1];

//                 }
//             }
//             input.reset_shape();
//             output.reset_shape();
//             return output;
//         }
// };

// #endif

#ifndef stft_H
#define stft_H
#include "cmath"
#include "fftw3.h"
#include "Matrix.hpp"
#include "feature.hpp"

// STFT Class
template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE, std::size_t FINAL_SIZE>
class stft{
    private:
        // Constants
        static constexpr std::size_t NUM_FRAMES = 1 + TOTAL_SAMPLES / HOP_SIZE;
        static constexpr std::size_t NUM_BINS = FFT_SIZE/2 + 1;
        static constexpr std::size_t FINAL = FINAL_SIZE;//max(max(NUM_FRAMES*NUM_BINS,TOTAL_SAMPLES),FINAL_SIZE);
        static constexpr std::size_t PAD_LEN = FFT_SIZE/2;
        //Matrix<T,FINAL,1> signal;
        Matrix<T,NUM_BINS,1> buffer1;
        Matrix<T,NUM_BINS,1> buffer2;
        Matrix<T,NUM_BINS,1> buffer3;
        Matrix<T,NUM_BINS,1> buffer4;
        Matrix<T,NUM_BINS,1> buffer5;
        Matrix<T,NUM_BINS,1> buffer6;
        Matrix<T,NUM_BINS,1> buffer7;
        Matrix<T,NUM_BINS,1> buffer8;
        Matrix<T,NUM_BINS,1> buffer9;

        // Private Methods
        void window_with_hann(int start,Matrix<T,FFT_SIZE,1>hann, const Matrix<T,FINAL,1> &signal, Matrix<T,FFT_SIZE,1> &output);
        Matrix<T,FFT_SIZE,1> hann_generator();
    public:
        // Public Methods
        //void setSignal(Matrix<T,TOTAL_SAMPLES,1>input);
        
        // Compute STFT Function
        void compute(const Matrix<T,FINAL,1> &signal, Matrix<T,NUM_FRAMES*NUM_BINS,1> &output){
            output.reshape(NUM_FRAMES,NUM_BINS);
            // Matrix<T,NUM_FRAMES,NUM_BINS> output;

            float in[FFT_SIZE];
            fftwf_complex out[NUM_BINS];

            fftwf_plan plan = fftwf_plan_dft_r2c_1d(FFT_SIZE,in,out,FFTW_ESTIMATE);

            Matrix<T,FFT_SIZE,1> hann = hann_generator();
            Matrix<T,FFT_SIZE,1> window;

            for(std::size_t i=0;i<NUM_FRAMES;i++){
                int start = static_cast<int>(i) * static_cast<int>(HOP_SIZE) - static_cast<int>(PAD_LEN);
                
                window_with_hann(start,hann,signal,window);

                for(std::size_t j=0;j<FFT_SIZE;j++){
                    in[j] = static_cast<float>(window[j][0]);
                }

                fftwf_execute(plan);

                // signal.reshape(NUM_FRAMES,NUM_BINS);
                for(std::size_t j=0;j<NUM_BINS;j++){
                    output[i][j] = std::sqrt(out[j][0]*out[j][0] + out[j][1]*out[j][1]);
                    // if(i>=7){
                    //     signal[i-9][j] = buffer9[j][0];
                    // }
                    // buffer9[j][0] = buffer8[j][0];
                    // buffer8[j][0] = buffer7[j][0];
                    // buffer7[j][0] = buffer6[j][0];
                    // buffer6[j][0] = buffer5[j][0];
                    // buffer5[j][0] = buffer4[j][0];
                    // buffer4[j][0] = buffer3[j][0];
                    // buffer3[j][0] = buffer2[j][0];
                    // buffer2[j][0] = buffer1[j][0];
                    // buffer1[j][0] = std::sqrt(out[j][0]*out[j][0] + out[j][1]*out[j][1]);
                }
                // signal.reset_shape();
            }
            
            // signal.reshape(NUM_FRAMES,NUM_BINS);
            for(std::size_t i=NUM_FRAMES-9;i<NUM_FRAMES;i++){
                for(std::size_t j=0;j<NUM_BINS;j++){
                    // signal[i][j] = buffer9[j][0];
                    // buffer9[j][0] = buffer8[j][0];
                    // buffer8[j][0] = buffer7[j][0];
                    // buffer7[j][0] = buffer6[j][0];
                    // buffer6[j][0] = buffer5[j][0];
                    // buffer5[j][0] = buffer4[j][0];
                    // buffer4[j][0] = buffer3[j][0];
                    // buffer3[j][0] = buffer2[j][0];
                    // buffer2[j][0] = buffer1[j][0];
                }
            }
            // for(std::size_t i=NUM_BINS;i<signal.cols();i++){
            //     for(std::size_t j=0;j<NUM_BINS;j++){
            //         signal[i][j] = 0;
            //     }
            // }
            // signal.reset_shape();

            fftwf_destroy_plan(plan);

        }
};

// Set Signal Function
/*
template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE>
void stft<T,TOTAL_SAMPLES,FFT_SIZE,HOP_SIZE>::setSignal(Matrix<T,TOTAL_SAMPLES,1> input){
    for(std::size_t i=0;i<TOTAL_SAMPLES;i++){
        signal[i][0] = input[i][0];
    }
}
*/

// Compute Window Function
template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE, std::size_t FINAL_SIZE>
void stft<T,TOTAL_SAMPLES,FFT_SIZE,HOP_SIZE,FINAL_SIZE>::window_with_hann(int start,Matrix<T,FFT_SIZE,1>hann, const Matrix<T,FINAL,1> &signal, Matrix<T,FFT_SIZE,1> &output){
    for(std::size_t i=0;i<FFT_SIZE;i++){
        int index = start + i;
        // while(index < 0 || index >= TOTAL_SAMPLES){
        //     if(index < 0)
        //         index = -index;
        //     else
        //         index = 2*TOTAL_SAMPLES - index - 2;
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

// Hann Generator Function
// Used to smooth the frequency domain
template <class T, std::size_t TOTAL_SAMPLES, std::size_t FFT_SIZE, std::size_t HOP_SIZE, std::size_t FINAL_SIZE>
Matrix<T,FFT_SIZE,1> stft<T,TOTAL_SAMPLES,FFT_SIZE,HOP_SIZE,FINAL_SIZE>::hann_generator(){
    Matrix<T,FFT_SIZE,1> output;
    T pi = static_cast<T>(3.14159265358979323846);
    for(std::size_t i=0;i<FFT_SIZE;i++){
        //output[i][0] = 1; // To Disable Hann uncomment this line. 
        output[i][0] = static_cast<T>(0.5) - static_cast<T>(0.5) * std::cos((static_cast<T>(2)*pi*i)/(FFT_SIZE-1));
    }
    return output;
}

// Convert to log thingy
template <class T, std::size_t N>
void log_thingy(Matrix<T,N,1> &input){
    T max_thingy = max_value(input);
    T zero_error = 1e-10;
    max_thingy = max_thingy + zero_error;
    for(std::size_t i=0;i<N;i++){
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
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T mag_total = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
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
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T band_1 = 0;
                for(std::size_t j=10;j<=22;j++){
                    band_1 = band_1 + input[i][j]*input[i][j];
                }
                T band_2 = 0;
                for(std::size_t j=26;j<=32;j++){
                    band_2 = band_2 + input[i][j]*input[i][j];
                }
                if(band_1 == 0) ber[i][0] = 0;
                else ber[i][0] = band_2/band_1;
            }
            input.reset_shape();
            return mean(ber);
        }
};

// Zoom STFT Class
template <class T, std::size_t N, std::size_t Input_row, std::size_t Input_col, std::size_t Output_row, std::size_t Output_col>
class zoom_stft{
    public:
        Matrix<T,Output_row*Output_col,1> output;
        // Zoom Function
        Matrix<T,Output_row*Output_col,1> zoom(Matrix<T,N,1> &input){
            input.reshape(Input_row,Input_col);
            output.reshape(Output_row,Output_col);
            T row_scale = static_cast<T>(Input_row-1) / (Output_row-1);
            T col_scale = static_cast<T>(Input_col-1) / (Output_col-1);
            for(std::size_t i=0;i<Output_row;i++){
                for(std::size_t j=0;j<Output_col;j++){
                    // std::size_t r0 = i*row_scale;
                    // T wr = r0 - std::floor(r0);
                    // r0 = std::floor(r0); // Floor Row
                    // std::size_t r1 = r0 + 1;// Ceil Row
                    // if(r1 >= Input_row) r1 = Input_row-1;

                    // std::size_t c0 = j*col_scale;
                    // T wc = c0 - std::floor(c0);
                    // c0 = std::floor(c0);// Floor Col
                    // std::size_t c1 = c0 + 1;// Ceil Col
                    // if(c1 >= Input_col) c1 = Input_col-1;
                    T r = i * row_scale;
                    std::size_t r0 = static_cast<std::size_t>(std::floor(r));
                    T wr = r - r0;
                    std::size_t r1 = r0 + 1;
                    if (r1 >= Input_row) r1 = Input_row - 1;

                    T c = j * col_scale;
                    std::size_t c0 = static_cast<std::size_t>(std::floor(c));
                    T wc = c - c0;
                    std::size_t c1 = c0 + 1;
                    if (c1 >= Input_col) c1 = Input_col - 1;

                    output[i][j] = (1-wr)*(1-wc)*input[r0][c0] + (1-wr)*wc*input[r0][c1] + wr*(1-wc)*input[r1][c0] + wr*wc*input[r1][c1];

                }
            }
            input.reset_shape();
            output.reset_shape();
            return output;
        }
};
#endif