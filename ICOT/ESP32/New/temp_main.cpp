#include <iostream>
#include <bits/stdc++.h>
#include "Matrix.hpp"
#include "disp.hpp"
#include "dwt.hpp"
using namespace std;

int main(){
    Matrix<float,10*10,1> s;
    s.reshape(10,10);
    int index = 0;
    for(int i=0;i<10;i++){
        for(int j=0;j<10;j++){
            s[i][j] = index++;
        }
    }
    disp(s);
    zoom_dwt<float,100,10,10,5,5> zoom_object;
    auto final = zoom_object.zoom(s);
    disp(final);
}