#ifndef dwt_H
#define dwt_H
#include <iostream>
#include "Matrix.hpp"

template <class T, std::size_t TOTAL_SAMPLES,std::size_t max_level>
class dwt{
    public:
        static constexpr int wavelet_size = 8;
        float filter_l[wavelet_size] = {
            2.303778133088965008632911830440708500016152482483092977910968e-01,
            7.148465705529156470899219552739926037076084010993081758450110e-01,
            6.308807679298589078817163383006152202032229226771951174057473e-01,
            -2.798376941685985421141374718007538541198732022449175284003358e-02,
            -1.870348117190930840795706727890814195845441743745800912057770e-01,
            3.084138183556076362721936253495905017031482172003403341821219e-02,
            3.288301166688519973540751354924438866454194113754971259727278e-02,
            -1.059740178506903210488320852402722918109996490637641983484974e-02
        }; // Low pass filter
        float filter_h[wavelet_size]; // High pass filter
        // High pass filter is same as low pass filter but reverse it and multiply each element with (-1)*n where n is index. (0 based indexing)

        static constexpr size_t compute_final_size(){
            size_t temp_final = 0;
            size_t temp_total = TOTAL_SAMPLES;
            size_t temp_level = max_level;
            size_t temp_D = 0;
            while(temp_level!=0){
                temp_final = 2*((temp_total+wavelet_size-1)/2) + temp_D;
                temp_D = temp_D + (temp_total+wavelet_size-1)/2;
                temp_total = (temp_total+wavelet_size-1)/2;
                temp_level--;
            }
            return temp_final;
        }
        static constexpr size_t final_size = compute_final_size();

        Matrix<T,final_size,1> signal;
        Matrix<T,(TOTAL_SAMPLES+wavelet_size-1)/2,1> A;
        Matrix<T,(TOTAL_SAMPLES+wavelet_size-1)/2,1> D;

        static constexpr int shift = wavelet_size/2 - 1;
        void compute_dwt(int level,size_t input,size_t current);
        void set_HighFilter();

    public:
        // Public Methods
        void setSignal(Matrix<T,TOTAL_SAMPLES,1>input);
        
        Matrix<T,final_size,1> compute(){
            // Set High Filter Wavelet
            set_HighFilter();
            compute_dwt(0,TOTAL_SAMPLES,final_size);
            return signal;
        }

};

// Set Signal Function
template <class T, std::size_t TOTAL_SAMPLES,std::size_t max_level>
void dwt<T,TOTAL_SAMPLES,max_level>::setSignal(Matrix<T,TOTAL_SAMPLES,1> input){
    for(size_t i=0;i<TOTAL_SAMPLES;i++){
        signal[i][0] = input[i][0];
    }
}

// Compute DWT Function
template <class T, std::size_t TOTAL_SAMPLES,std::size_t max_level>
void dwt<T,TOTAL_SAMPLES,max_level>::compute_dwt(int level,size_t input,size_t current){
    if(level==max_level) return;
    if(input==0) return;
    size_t limit = input;
    input = (input+wavelet_size-1)/2;
    for(size_t i=0;i<input;i++){
        A[i][0] = 0;
        D[i][0] = 0;
    }
    for(int i=1;i<(int)input*2+1;i=i+2){
        for(int j=0;j<wavelet_size;j++){
            int index = i-j;
            if(index<0) A[i/2][0] = A[i/2][0] + filter_l[wavelet_size-j-1]*signal[-index-1][0];
            else if(index>=0 && index<limit) A[i/2][0] = A[i/2][0] + filter_l[wavelet_size-j-1]*signal[index][0];
            else A[i/2][0] = A[i/2][0] + filter_l[wavelet_size-j-1]*signal[2*limit-1-index][0];

            if(index<0) D[i/2][0] = D[i/2][0] + filter_h[wavelet_size-j-1]*signal[-index-1][0];
            else if(index>=0 && index<limit) D[i/2][0] = D[i/2][0] + filter_h[wavelet_size-j-1]*signal[index][0];
            else D[i/2][0] = D[i/2][0] + filter_h[wavelet_size-j-1]*signal[2*limit-1-index][0];
        }
    }
    for(size_t i=0;i<input;i++){
        signal[i][0] = A[i][0];
        signal[current-(input-i)][0] = D[i][0];
    }
    compute_dwt(level+1,input,current-input);
}

template <class T, std::size_t TOTAL_SAMPLES,std::size_t max_level>
void dwt<T,TOTAL_SAMPLES,max_level>::set_HighFilter(){
    float sign = 1;
    for(int i=0;i<wavelet_size;i++){
        filter_h[i] = filter_l[wavelet_size - i - 1] * sign;
        sign = -1*sign;
    }
}

#endif