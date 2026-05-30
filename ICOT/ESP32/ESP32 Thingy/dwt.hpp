#ifndef dwt_H
#define dwt_H
#include <iostream>
#include "Matrix.hpp"
#include "feature.hpp"
#include "esp_task_wdt.h"
#define YIELD() delay(1)

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
        
        void compute(Matrix<T,final_size,1> &signal){
            // Set High Filter Wavelet
            set_HighFilter();
            compute_dwt(0,TOTAL_SAMPLES,final_size,signal);
            signal.reset_shape();
            YIELD(); for(std::size_t i=0;i<signal.rows();i++) signal[i][0] = std::abs(signal[i][0]);
        }

};

// Set Signal Function
/*
template <class T, std::size_t TOTAL_SAMPLES,std::size_t max_level>
void dwt<T,TOTAL_SAMPLES,max_level>::setSignal(Matrix<T,TOTAL_SAMPLES,1> input){
    YIELD(); for(size_t i=0;i<TOTAL_SAMPLES;i++){
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
    YIELD(); for(size_t i=0;i<input;i++){
        //A[i][0] = 0;
        D[i][0] = 0;
    }
    YIELD(); for(size_t i=0;i<20;i++) buffer[i][0] = 0;
    int buffer_corrector = 0;
    YIELD(); for(int i=1;i<(int)input*2+1;i=i+2){
        T A = 0;
        YIELD(); for(int j=0;j<wavelet_size;j++){
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
    YIELD(); for(size_t i=std::max(static_cast<int>(0),static_cast<int>(input)-20);i<input;i++){
        signal[i][0] = cycle(0);
    }
    YIELD(); for(size_t i=0;i<input;i++){
        //signal[i][0] = A[i][0];
        signal[current-(input-i)][0] = D[i][0];
    }
    //YIELD(); for(size_t i=0;i<20;i++) buffer[i][0] = 0;
    compute_dwt(level+1,input,current-input,signal);
}

template <class T, std::size_t TOTAL_SAMPLES,std::size_t max_level>
void dwt<T,TOTAL_SAMPLES,max_level>::set_HighFilter(){
    float sign = 1;
    YIELD(); for(int i=0;i<wavelet_size;i++){
        filter_h[i] = filter_l[wavelet_size - i - 1] * sign;
        sign = -1*sign;
    }
}

template <class T, std::size_t TOTAL_SAMPLES,std::size_t max_level>
T dwt<T,TOTAL_SAMPLES,max_level>::cycle(T data){
    T return_data = buffer[0][0];
    YIELD(); for(size_t i=1;i<20;i++){
        buffer[i-1][0] = buffer[i][0];
    }
    buffer[19][0] = data;
    return return_data;
}

// Zoom DWT Class
template <class T, std::size_t N, std::size_t Input_row, std::size_t Input_col, std::size_t Output_row, std::size_t Output_col>
class zoom_dwt{
    private:
        T temp_maximum = 1e-10;
    public:
        Matrix<T,Output_row*Output_col,1> output;


        // Padding Simulator Function
        T get_coeff(Matrix<T,N,1> &input, int row, int col){
            // if(row < 0 || row >= Input_row || col < 0 || col >= Input_col)
            //     return 0;
            // return input[row*10+col][0];
            // Padding thingys for dwt scalogram since each level has a different size.
            // static constexpr std::size_t row_offsets[7] = {0, 257, 764, 1770, 3776, 7782, 15787};
            // static constexpr std::size_t row_lengths[7] = {256, 763, 1769, 3775, 7781, 15786, 32000}; // Recheck this row_lengths again.
            static constexpr std::size_t row_lengths[7] = {506, 506, 1006, 2006, 4006, 8005, 16003};
            static constexpr std::size_t row_offsets[7] = {0, 506, 1012, 2018, 4024, 8030, 16035};
            
            // if(row >=Input_row || row >=7) return static_cast<T>(20*std::log10(1e-10/temp_maximum));
            if(row >=Input_row || row >=7) return -80;
            // if(col >= row_lengths[row]) return static_cast<T>(20*std::log10(1e-10/temp_maximum));
            if(col >= row_lengths[row]) return -80;
            return input[row_offsets[row] + col][0];
        }
        // Zoom Function
        Matrix<T,Output_row*Output_col,1> zoom(Matrix<T,N,1> &input){
            input.reset_shape();
            output.reshape(Output_col,Output_row);
            
            temp_maximum = max_value(input);
            log_thingy(input);

            // T row_scale = static_cast<T>(Input_row) / static_cast<T>(Output_row);
            // T col_scale = static_cast<T>(Input_col) / static_cast<T>(Output_col);
            // YIELD(); for(std::size_t i=0;i<Output_row;i++){
            //     T r = (i+0.5)*row_scale-0.5;
            //     int r0 = std::max((int)std::floor(r),(0));
            //     int r1 = std::min(r0+1,(int)Input_row-1);
            //     T wr = r - r0;
            //     YIELD(); for(std::size_t j=0;j<Output_col;j++){
            //         T c = (j+0.5)*col_scale-0.5;
            //         int c0 = std::max((int)std::floor(c),(0));
            //         int c1 = std::min(c0+1,(int)Input_col-1);
            //         T wc = c - c0;           
            //         output[i][j] = (1-wr)*(1-wc)*get_coeff(input,r0,c0) + (1-wr)*wc*get_coeff(input,r0,c1) + wr*(1-wc)*get_coeff(input,r1,c0) + wr*wc*get_coeff(input,r1,c1);

            //     }
            // }

            T row_scale = (T)(Output_row-1) / (T)(Input_row-1);
            T col_scale = (T)(Output_col-1) / (T)(Input_col-1);

            YIELD(); for(size_t i=0;i<Output_row;i++){

                T r = (i)/row_scale;

                int r0 = floor(r);
                int r1 = r0 + 1;

                T wr = r - r0;

                YIELD(); for(size_t j=0;j<Output_col;j++){

                    T c = (j)/col_scale;

                    int c0 = floor(c);
                    int c1 = c0 + 1;

                    T wc = c - c0;

                    T v00 = get_coeff(input,r0,c0);
                    T v01 = get_coeff(input,r0,c1);
                    T v10 = get_coeff(input,r1,c0);
                    T v11 = get_coeff(input,r1,c1);

                    output[j][i] =
                        (1-wr)*(1-wc)*v00 +
                        (1-wr)*wc*v01 +
                        wr*(1-wc)*v10 +
                        wr*wc*v11;
                }
            }
            input.reset_shape();
            // output.reshape(Output_row,Output_col);
            // YIELD(); for(std::size_t i=0;i<Output_row;i++){
            //     YIELD(); for(std::size_t j=i+1;j<Output_col;j++){
            //         T temp = output[i][j];
            //         output[i][j] = output[j][i];
            //         output[j][i] = temp;
            //     }
            // }
            // output.reset_shape();
            return output;
        }
};
#endif