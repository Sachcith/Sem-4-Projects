#ifndef disp_H
#define disp_H
#include <cstddef>
#include <iostream>
#include "Matrix.hpp"

template<class T, std::size_t R, std::size_t C>
void disp(Matrix<T,R,C> x){
    std::size_t tempR = x.rows();
    std::size_t tempC = x.cols();
    for(int i=0;i<tempR;i++){
        for(int j=0;j<tempC;j++){
            std::cout<<x[i][j]<<" ";
        }
        std::cout<<endl;
    }
    std::cout<<endl;
}
#endif