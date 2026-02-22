#ifndef activation_H
#define activation_H
#include <cstddef>
#include <cmath>
#include "Matrix.hpp"

template<class T, std::size_t R, std::size_t C>
Matrix<T,R,C> relu(Matrix<T,R,C> input){
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            if(input[i][j]<0) input[i][j] = 0;
        }
    }
    return input;
}

template<class T, std::size_t OUT>
Matrix<T,OUT,1> softmax(Matrix<T,OUT,1> input){
    T max_val = input[0][0];
    for(int i=1;i<OUT;i++){
        if(max_val<input[i][0]) max_val = input[i][0];
    }
    T total = 0;
    for(int i=0;i<OUT;i++){
        input[i][0] = static_cast<T>(exp(static_cast<float>(input[i][0] - max_val)));
        total = total + input[i][0];
    }
    for(int i=0;i<OUT;i++){
        input[i][0] = input[i][0]/total;
    }
    return input;
}
#endif