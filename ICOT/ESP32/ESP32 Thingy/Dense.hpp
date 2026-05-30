#ifndef Dense_H
#define Dense_H
#include "Matrix.hpp"
#include <cstddef>
#include <pgmspace.h>
#include "esp_task_wdt.h"
// #define YIELD() esp_task_wdt_reset()

template<class T, std::size_t OUT, std::size_t IN>
class Dense{
    public:
        void forward(Matrix<T,IN,1> &input,const float* weight, const float* bias, Matrix<T,OUT,1> &output);
};

template <class T, std::size_t OUT, std::size_t IN>
void Dense<T,OUT,IN>::forward(Matrix<T,IN,1> &input, const float* weight, const float* bias, Matrix<T,OUT,1> &output){
    YIELD(); for(int i=0;i<OUT;i++){
        output[i][0] = 0;
        YIELD(); for(int k=0;k<IN;k++){
            output[i][0] = output[i][0] + pgm_read_float(&weight[k*OUT + i])*input[k][0];
        }
        output[i][0]  = output[i][0] + pgm_read_float(&bias[i]);
    }
}
#endif