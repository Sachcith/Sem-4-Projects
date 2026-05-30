#include <iostream>
#include <bits/stdc++.h>
using namespace std;

#include "Matrix.hpp"

template <class T,std::size_t N>
Matrix<T,N,1> unshuffle(Matrix<T,N,1> &input){

}

int main(){
    constexpr size_t n = 10;
    Matrix<float,n,1> x;
    float tempx[n] = {1,10,2,9,3,8,4,7,5,6};

    // 1,6,2,7,3,8,4,9,5,10
    // 1,8,2,7,3,6,4,9,5,10
    // 1,9,2,7,3,6,4,8,5,10
    // 1,5,2,7,3,6,4,8,9,10
    // 1,3,2,7,5,6,4,8,9,10
    // 1,2,3,7,5,6,4,8,9,10

    //
    // 
    // 
}