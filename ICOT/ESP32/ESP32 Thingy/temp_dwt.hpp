#ifndef dwt_H
#define dwt_H
#include "Matrix.hpp"

template <class T, std::size_t TOTAL_SAMPLES>
class dwt{
    private:
        static constexpr int wavelet_size = 8;
        float filter_l[wavelet_size] = {0.2304,0.7148,0.6309,-0.0280,-0.1870,0.0308,0.0329,-0.0106}; // Low pass filter
        float filter_h[wavelet_size]; // High pass filter
        // High pass filter is same as low pass filter but reverse it and multiply each element with (-1)*n where n is index. (0 based indexing)

        Matrix<T,TOTAL_SAMPLES,1> signal;
        Matrix<T,TOTAL_SAMPLES/2,1> A;
        Matrix<T,TOTAL_SAMPLES/2,1> D;

        int stop_level;
        static constexpr int shift = wavelet_size/2 - 1;
        void compute_dwt(int level,size_t input);
        void set_HighFilter();

    public:
        // Public Methods
        void setSignal(Matrix<T,TOTAL_SAMPLES,1>input);
        Matrix<T,TOTAL_SAMPLES,1> compute(int level);

};

// Set Signal Function
template <class T, std::size_t TOTAL_SAMPLES>
void dwt<T,TOTAL_SAMPLES>::setSignal(Matrix<T,TOTAL_SAMPLES,1> input){
    for(size_t i=0;i<TOTAL_SAMPLES;i++){
        signal[i][0] = input[i][0];
    }
}

// Compute DWT Function
template <class T, std::size_t TOTAL_SAMPLES>
void dwt<T,TOTAL_SAMPLES>::compute_dwt(int level,size_t input){
    if(level==stop_level) return;
    if(input==0) return;
    for(size_t i=0;i<input/2;i++){
        A[i][0] = 0;
        D[i][0] = 0;
    }
    for(size_t i=0;i<input;i=i+2){
        for(size_t j=0;j<wavelet_size;j++){
            if(i+j<input) A[i/2][0] = A[i/2][0] + filter_l[j]*signal[i+j][0];
            else A[i/2][0] = A[i/2][0] + filter_l[j]*signal[input-1-j][0];

            if(i+j<input) D[i/2][0] = D[i/2][0] + filter_h[j]*signal[i+j][0];
            else D[i/2][0] = D[i/2][0] + filter_h[j]*signal[input-1-j][0];
        }
    }
    for(size_t i=0;i<input/2;i++){
        signal[i][0] = A[i][0];
        signal[i+input/2][0] = D[i][0];
    }
    compute_dwt(level+1,input/2);
}

template <class T, std::size_t TOTAL_SAMPLES>
Matrix<T,TOTAL_SAMPLES,1> dwt<T,TOTAL_SAMPLES>::compute(int level){
    stop_level = level;
    // Set High Filter Wavelet
    set_HighFilter();
    compute_dwt(0,TOTAL_SAMPLES);
    return signal;
};

template <class T, std::size_t TOTAL_SAMPLES>
void dwt<T,TOTAL_SAMPLES>::set_HighFilter(){
    float sign = 1;
    for(int i=0;i<wavelet_size;i++){
        filter_h[i] = filter_l[wavelet_size - i - 1] * sign;
        sign = -1*sign;
    }
}

#endif