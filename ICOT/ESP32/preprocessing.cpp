#include <iostream>
#include <bits/stdc++.h>
using namespace std;

#include "Matrix.hpp"
#include "disp.hpp"
#include "Dense.hpp"
#include "activation.hpp"
#include "stft.hpp"
#include "dwt.hpp"

int main(){
    const size_t total_size = 64; //8;

    const int max_level = 3;
    static dwt<float,total_size,max_level> x;
    cout<<x.final_size<<endl;

    Matrix<float,total_size,1> y;
    //float tempy[total_size] = {3,1,0,4,8,6,9,2};
    //float tempy[total_size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float tempy[total_size];
    for(int i=0;i<total_size;i++) tempy[i] = i+1;

    for(size_t i=0;i<total_size;i++){
        y[i][0] = tempy[i];
    }

    x.setSignal(y);
    disp(x.compute());


    /*
    const size_t total_size = 16;
    const size_t fft_size = 8;
    const size_t hop_size = 4;

    stft<float,total_size,fft_size,hop_size> x;

    Matrix<float,total_size,1> y;
    float tempy[total_size] = {
        0,1,0,-1, 0,1,0,-1,
        0,1,0,-1, 0,1,0,-1
    };
    
    for(size_t i=0;i<total_size;i++){
        y[i][0] = tempy[i];
    }
    
    x.setSignal(y);
    auto temp = x.compute();
    disp(temp);
    */


    /*
    Matrix<float,3,3> x;
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            if(i==j) x[i][j] = 1;
            else x[i][j] = 0;
        }
    }
    disp(x);
    Dense<float,3,3> d;
    d.setWeight(x);
    Matrix<float,3,1> y;
    for(int i=0;i<3;i++){
        y[i][0] = -i;
    }
    auto temp = d.forward(y);
    disp(temp);
    temp = relu(temp);
    disp(temp);

    //Matrix<float,3,1> y;
    y[0][0] = 2.0f;
    y[1][0] = 1.0f;
    y[2][0] = 0.1f;
    disp(softmax(y));
    */
}