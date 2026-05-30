#ifndef Dense_H
#define Dense_H
#include "Matrix.hpp"
#include <cstddef>

template<class T, std::size_t OUT, std::size_t IN>
class Dense{
    public:
        Matrix<T,OUT,1> forward(Matrix<T,IN,1> &input,const float* weight, const float* bias);
};

template <class T, std::size_t OUT, std::size_t IN>
Matrix<T,OUT,1> Dense<T,OUT,IN>::forward(Matrix<T,IN,1> &input, const float* weight, const float* bias){
    Matrix<T,OUT,1> output;
    for(int i=0;i<OUT;i++){
        output[i][0] = 0;
        for(int k=0;k<IN;k++){
            output[i][0] = output[i][0] + weight[i*IN + k]*input[k][0];
        }
        output[i][0]  = output[i][0] + bias[i];
    }
    return output;
}
#endif