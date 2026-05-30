#ifndef audio_cleaner_H
#define audio_cleaner_H
#include <cstddef>
#include "Matrix.hpp"
#include "feature.hpp"
#include "esp_task_wdt.h"
#define YIELD() delay(1)

// Pre Emphasis Function
// To hilight sudden changes in the signal
template <class T,std::size_t N>
void apply_pre_emphasis(Matrix<T,N,1> &input, T coeff){
    T buffer = input[0][0];
    T temp = 0;
    YIELD(); for(std::size_t i=1;i<N;i++){
        temp = input[i][0];
        input[i][0] = input[i][0] - coeff * buffer;
        buffer = temp;
    }
}
// Band Pass Filter Function
// Using Butterworth Filter; Low = 100Hz; High = 1000Hz
// order = 5
template <class T,std::size_t N>
void apply_bandpass_filter(Matrix<T,N,1> &input_nopad){
    struct Biquad{
        T b0,b1,b2;
        T a1,a2;

        T w1 = 0;
        T w2 = 0;

        T process(T x){
            T w0 = x - a1*w1 - a2*w2;

            T y  = b0*w0 + b1*w1 + b2*w2;

            w2 = w1;
            w1 = w0;

            return y;
        }
    };
    T sos[5][6] = {
        { 0.00010213196630180957, 0.00020426393260361913, 0.00010213196630180957, 1.0, -1.5034659413638414, 0.6000651573494533 },
        { 1.0, 2.0, 1.0, 1.0, -1.688079584963568, 0.8233751401399853 },
        { 1.0, 0.0, -1.0, 1.0, -1.6837556388140702, 0.6969611825284777 },
        { 1.0, -2.0, 1.0, 1.0, -1.9354700569439443, 0.9373685406898521 },
        { 1.0, -2.0, 1.0, 1.0, -1.9782489881358187, 0.9798163940943425 }
    };
    constexpr std::size_t pad = 2*(3*(11-1)); // 2 * (3 * (max(len(a),len(b))-1))
    Matrix<T,N+pad,1> input;
    YIELD(); for(std::size_t i=30;i<N+30;i++) input[i][0] = input_nopad[i-30][0];
    // YIELD(); for(std::size_t i=0;i<30;i++){
    //     input[i][0] = input_nopad[pad/2 - i][0];
    //     input[N + pad/2 + i][0] = input_nopad[N - 2 - i][0];
    // }
    YIELD(); for(size_t i=0;i<30;i++){
        input[i][0] = 2*input_nopad[0][0] - input_nopad[i+1][0];
    }
    YIELD(); for(size_t i=0;i<30;i++){
        input[N+30+i][0] = 2*input_nopad[N-1][0] - input_nopad[N-2-i][0];
    }
    Biquad biquad1[5];
    YIELD(); for(int i=0;i<5;i++){
        biquad1[i].b0 = sos[i][0];
        biquad1[i].b1 = sos[i][1];
        biquad1[i].b2 = sos[i][2];
        biquad1[i].a1 = sos[i][4];
        biquad1[i].a2 = sos[i][5];
    }
    YIELD(); for(std::size_t i=0;i<N+60;i++){
        YIELD(); for(std::size_t j=0;j<5;j++){
            input[i][0] = biquad1[j].process(input[i][0]);
        }
    }
    T temp;
    YIELD(); for(std::size_t i=0;i<(N+60)/2;i++){
        temp = input[i][0];
        input[i][0] = input[(N+60)-i-1][0];
        input[(N+60)-i-1][0] = temp;
    }
    Biquad biquad2[5];
    YIELD(); for(int i=0;i<5;i++){
        biquad2[i].b0 = sos[i][0];
        biquad2[i].b1 = sos[i][1];
        biquad2[i].b2 = sos[i][2];
        biquad2[i].a1 = sos[i][4];
        biquad2[i].a2 = sos[i][5];
    }
    YIELD(); for(std::size_t i=0;i<N+60;i++){
        YIELD(); for(std::size_t j=0;j<5;j++){
            input[i][0] = biquad2[j].process(input[i][0]);
        }
    }
    YIELD(); for(std::size_t i=0;i<(N+60)/2;i++){
        temp = input[i][0];
        input[i][0] = input[(N+60)-i-1][0];
        input[(N+60)-i-1][0] = temp;
    }
    YIELD(); for(std::size_t i=0;i<N;i++){
        input_nopad[i][0] = input[i+pad/2][0];
    }
}

// Queue simulator for Butterworth Filter
// template <class T, std::size_t N>
// void cycle(Matrix<T,N,1> &input,T data){
//     int n = input.rows();
//     YIELD(); for(int i=n-1;i>=1;i--){
//         input[i][0] = input[i-1][0];
//     }
//     input[0][0] = data;
// }

// Normalize Function using RMS
template <class T,std::size_t N>
void normalize_rms(Matrix<T,N,1> &input){
    T rms = RMS(input);
    if(rms<=1e-6) return;
    YIELD(); for(std::size_t i=0;i<N;i++){
        input[i][0] = input[i][0]/rms;
    }
}
#endif