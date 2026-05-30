#ifndef dwt_H
#define dwt_H
#include <iostream>
#include "Matrix.hpp"

template <class T, std::size_t TOTAL_SAMPLES,std::size_t max_level>
class dwt{
    public:
        static constexpr int wavelet_size = 8;
        T filter_l[wavelet_size] = {
            2.303778133088965008632911830440708500016152482483092977910968e-01,
            7.148465705529156470899219552739926037076084010993081758450110e-01,
            6.308807679298589078817163383006152202032229226771951174057473e-01,
            -2.798376941685985421141374718007538541198732022449175284003358e-02,
            -1.870348117190930840795706727890814195845441743745800912057770e-01,
            3.084138183556076362721936253495905017031482172003403341821219e-02,
            3.288301166688519973540751354924438866454194113754971259727278e-02,
            -1.059740178506903210488320852402722918109996490637641983484974e-02
        }; // Low pass filter
        T filter_h[wavelet_size]; // High pass filter
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

        //Matrix<T,final_size,1> signal;
        Matrix<T,(TOTAL_SAMPLES+wavelet_size-1)/2,1> A;
        Matrix<T,(TOTAL_SAMPLES+wavelet_size-1)/2,1> D;

        Matrix<T,20,1> buffer;
        T cycle(T data);

        static constexpr int shift = wavelet_size/2 - 1;
        void compute_dwt(int level,size_t input,size_t current, Matrix<T,final_size,1> &signal);
        void set_HighFilter();

    public:
        // Public Methods
        // void setSignal(Matrix<T,TOTAL_SAMPLES,1>input);
        
        Matrix<T,final_size,1> compute(Matrix<T,final_size,1> &signal){
            // Set High Filter Wavelet
            set_HighFilter();
            compute_dwt(0,TOTAL_SAMPLES,final_size,signal);
            return signal;
        }

};

// Set Signal Function
/*
template <class T, std::size_t TOTAL_SAMPLES,std::size_t max_level>
void dwt<T,TOTAL_SAMPLES,max_level>::setSignal(Matrix<T,TOTAL_SAMPLES,1> input){
    for(size_t i=0;i<TOTAL_SAMPLES;i++){
        signal[i][0] = input[i][0];
    }
}
*/

// Compute DWT Function
template <class T, std::size_t TOTAL_SAMPLES,std::size_t max_level>
void dwt<T,TOTAL_SAMPLES,max_level>::compute_dwt(int level,size_t input,size_t current, Matrix<T,final_size,1> &signal){
    if(level==max_level) return;
    if(input==0) return;
    size_t limit = input;
    input = (input+wavelet_size-1)/2;
    for(size_t i=0;i<input;i++){
        //A[i][0] = 0;
        D[i][0] = 0;
    }
    int buffer_corrector = 0;
    for(int i=1;i<(int)input*2+1;i=i+2){
        T A = 0;
        for(int j=0;j<wavelet_size;j++){
            int index = i-j;
            if(index<0) A = A + filter_l[wavelet_size-j-1]*signal[-index-1][0];
            else if(index>=0 && index<limit) A = A + filter_l[wavelet_size-j-1]*signal[index][0];
            else A = A + filter_l[wavelet_size-j-1]*signal[2*limit-1-index][0];

            if(index<0) D[i/2][0] = D[i/2][0] + filter_h[wavelet_size-j-1]*signal[-index-1][0];
            else if(index>=0 && index<limit) D[i/2][0] = D[i/2][0] + filter_h[wavelet_size-j-1]*signal[index][0];
            else D[i/2][0] = D[i/2][0] + filter_h[wavelet_size-j-1]*signal[2*limit-1-index][0];
        }
        if((i/2)>=20){
            signal[(i/2)-20][0] = cycle(A);
        }
        else{ 
            T garbage = cycle(A);
            buffer_corrector++;
        }
    }
    while(buffer_corrector!=20){
        cycle(0);
        buffer_corrector++;
    }
    for(size_t i=max(static_cast<int>(0),static_cast<int>(input)-20);i<input;i++){
        signal[i][0] = cycle(0);
    }
    for(size_t i=0;i<input;i++){
        //signal[i][0] = A[i][0];
        signal[current-(input-i)][0] = D[i][0];
    }
    //for(size_t i=0;i<20;i++) buffer[i][0] = 0;
    compute_dwt(level+1,input,current-input,signal);
}

template <class T, std::size_t TOTAL_SAMPLES,std::size_t max_level>
void dwt<T,TOTAL_SAMPLES,max_level>::set_HighFilter(){
    float sign = 1;
    for(int i=0;i<wavelet_size;i++){
        filter_h[i] = filter_l[wavelet_size - i - 1] * sign;
        sign = -1*sign;
    }
}

template <class T, std::size_t TOTAL_SAMPLES,std::size_t max_level>
T dwt<T,TOTAL_SAMPLES,max_level>::cycle(T data){
    T return_data = buffer[0][0];
    for(size_t i=1;i<20;i++){
        buffer[i-1][0] = buffer[i][0];
    }
    buffer[19][0] = data;
    return return_data;
}

// Zoom DWT Class
template <class T, std::size_t N, std::size_t Input_row, std::size_t Input_col, std::size_t Output_row, std::size_t Output_col>
class zoom_dwt{
    public:
        Matrix<T,Output_row,Output_col> output;


        // Padding Simulator Function
        T get_coeff(Matrix<T,N,1> &input, std::size_t row, std::size_t col){

            // Padding thingys for dwt scalogram since each level has a different size.
            static constexpr std::size_t row_offsets[7] = {0, 257, 764, 1770, 3776, 7782, 15787};
            static constexpr std::size_t row_lengths[7] = {256, 763, 1769, 3775, 7781, 15786, 32000}; // Recheck this row_lengths again.

            if(col >= row_lengths[row]) return static_cast<T>(0.0);
            return input[row_offsets[row] + col];
        }
        // Zoom Function
        Matrix<T,Output_row,Output_col> zoom(Matrix<T,N,1> &input){
            //input.reshape(Input_row,Input_col);
            T row_scale = static_cast<T>(Input_row-1) / (Output_row-1);
            T col_scale = static_cast<T>(Input_col-1) / (Output_col-1);
            for(std::size_t i=0;i<Output_row;i++){
                for(std::size_t j=0;j<Output_col;j++){
                    std::size_t r0 = static_cast<std::size_t>(i*row_scale);
                    T wr = r0 - std::floor(r0);
                    r0 = std::floor(r0); // Floor Row
                    std::size_t r1 = r0 + 1;// Ceil Row
                    if(r1 >= Input_row) r1 = Input_row-1;

                    std::size_t c0 = static_cast<std::size_t>(j*col_scale);
                    T wc = c0 - std::floor(c0);
                    c0 = std::floor(c0);// Floor Col
                    std::size_t c1 = c0 + 1;// Ceil Col
                    if(c1 >= Input_col) c1 = Input_col-1;

                    output[i][j] = (1-wr)*(1-wc)*get_coeff(input,r0,c0) + (1-wr)*wc*get_coeff(input,r0,c1) + wr*(1-wc)*get_coeff(input,r1,c0) + wr*wc*get_coeff(input,r1,c1);

                }
            }
            input.reset_shape();
            return output;
        }
};
#endif