#ifndef Dense_H
#define Dense_H
#include "Matrix.hpp"
#include <cstddef>

template<class T, std::size_t OUT, std::size_t IN>
class Dense{
    public:
        Matrix<T,OUT,IN> weight; // size = output * input
        Matrix<T,OUT,1> bias; // size = output * 1
    public:
        void setWeight(Matrix<T,OUT,IN> weight_pointer);
        void setBias(Matrix<T,OUT,1>);
        Matrix<T,OUT,1> forward(Matrix<T,IN,1> input);
};

template <class T, std::size_t OUT, std::size_t IN>
void Dense<T,OUT,IN>::setWeight(Matrix<T,OUT,IN> weight_input){
    for(int i=0;i<OUT;i++){
        for(int j=0;j<IN;j++){
            weight[i][j] = weight_input[i][j];
        }
    }
}

template <class T, std::size_t OUT, std::size_t IN>
void Dense<T,OUT,IN>::setBias(Matrix<T,OUT,1> bias_input){
    for(int i=0;i<OUT;i++){
        bias[i][0] = bias_input[i][0];
    }
}

template <class T, std::size_t OUT, std::size_t IN>
Matrix<T,OUT,1> Dense<T,OUT,IN>::forward(Matrix<T,IN,1> input){
    Matrix<T,OUT,1> output;
    for(int i=0;i<OUT;i++){
        output[i][0] = 0;
        for(int k=0;k<IN;k++){
            output[i][0] = output[i][0] + weight[i][k]*input[k][0];
        }
        output[i][0]  = output[i][0] + bias[i][0];
    }
    return output;
}
#endif