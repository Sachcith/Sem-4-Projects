#ifndef disp_H
#define disp_H
#include <cstddef>
#include <iostream>
#include "Matrix.hpp"

template<class T, std::size_t R, std::size_t C>
void disp(Matrix<T,R,C> x){
    for(int i=0;i<R;i++){
        for(int j=0;j<C;j++){
            std::cout<<x[i][j]<<" ";
        }
        std::cout<<endl;
    }
    std::cout<<endl;
}
#endif