#ifndef audio_cleaner_H
#define audio_cleaner_H
#include <cstddef>
#include "Matrix.hpp"
#include "feature.hpp"




// Pre Emphasis Function
// To hilight sudden changes in the signal
template <class T,std::size_t N>
void apply_pre_emphasis(Matrix<T,N,1> &input, T coeff){
    T buffer = input[0];
    T temp = 0;
    for(std::size_t i=1;i<N;i++){
        temp = input[i];
        input[i] = input[i] - coeff * buffer;
        buffer = temp;
    }
}


// Band Pass Filter Function
// Using Butterworth Filter; Low = 100Hz; High = 1000Hz
// order = 5
template <class T,std::size_t N>
void apply_bandpass_filter(Matrix<T,N,1> &input){
    T b[11] = {0.00010213196630180957,0.0,-0.0005106598315090478,0.0,0.0010213196630180956,0.0,-0.0010213196630180956,0.0,0.0005106598315090478,0.0,-0.00010213196630180957};
    T a[11] = {1.0,-8.789020210221242,34.85874005582234,-82.1727024950249,127.5140707982069,-136.11970915659327,101.23772690984748,-51.802341246568744,17.45345203175097,-3.496488018167832,0.3162713314618602};
    Matrix<T,11,1> xbuffer;
    Matrix<T,11,1> ybuffer;
    for(int i=0;i<11;i++){
        xbuffer[i][0] = 0;
        ybuffer[i][0] = 0;
    }
    for(std::size_t i=0;i<N;i++){
        cycle(xbuffer,input[i][0]);
        input[i][0] = 0;
        for(std::size_t j=0;j<11;j++){
            input[i][0] = input[i][0] + b[j]*xbuffer[j][0];
        }
        for(std::size_t j=0;j<10;j++){
            input[i][0] = input[i][0] - a[j+1]*ybuffer[j][0];
        }
        input[i][0] = input[i][0]/a[0];
        cycle(ybuffer,input[i][0]);
    }
    T temp;
    for(std::size_t i=0;i<N/2;i++){
        temp = input[i][0];
        input[i][0] = input[N-i-1][0];
        input[N-i-1][0] = temp;
    }


    for(int i=0;i<11;i++){
        xbuffer[i][0] = 0;
        ybuffer[i][0] = 0;
    }
    for(std::size_t i=0;i<N;i++){
        cycle(xbuffer,input[i][0]);
        input[i][0] = 0;
        for(std::size_t j=0;j<11;j++){
            input[i][0] = input[i][0] + b[j]*xbuffer[j][0];
        }
        for(std::size_t j=0;j<10;j++){
            input[i][0] = input[i][0] - a[j+1]*ybuffer[j][0];
        }
        input[i][0] = input[i][0]/a[0];
        cycle(ybuffer,input[i][0]);
    }
    for(std::size_t i=0;i<N/2;i++){
        temp = input[i][0];
        input[i][0] = input[N-i-1][0];
        input[N-i-1][0] = temp;
    }
}

// Queue simulator for Butterworth Filter
template <class T, std::size_t N>
void cycle(Matrix<T,N,1> &input,T data){
    int n = input.rows();
    for(int i=n-1;i>=1;i--){
        input[i][0] = input[i-1][0];
    }
    input[0][0] = data;
}

// Normalize Function using RMS
template <class T,std::size_t N>
void normalize_rms(Matrix<T,N,1> &input){
    T rms = RMS(input);
    if(rms==0) return;
    for(std::size_t i=0;i<N;i++){
        input[i][0] = input[i][0]/rms;
    }
}
#endif