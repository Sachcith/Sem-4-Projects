#include <iostream>
#include <bits/stdc++.h>
using namespace std;

#include "mlp_model_weights.cc"
#include "Dense_float.hpp"
#include "Matrix.hpp"
#include "activation.hpp"
#include "disp.hpp"

int main(){
    Matrix<float,70,1>final_features;

    Dense<float,64,70> hidden_1;
    Dense<float,32,64> hidden_2;
    Dense<float,6,32> hidden_3;

    auto h1_output = hidden_1.forward(final_features,mlp_weights::hidden_1_weights,mlp_weights::hidden_1_bias);
    relu(h1_output);
    auto h2_output = hidden_2.forward(h1_output,mlp_weights::hidden_2_weights,mlp_weights::hidden_2_bias);
    relu(h2_output);
    auto h3_output = hidden_3.forward(h2_output,mlp_weights::student_output_weights,mlp_weights::student_output_bias);
    softmax(h3_output);

    float temp_final_for_answer = -1;
    size_t final_answer;
    for(size_t i=0;i<h3_output.rows();i++){
        if(temp_final_for_answer<h3_output[i][0]){
            temp_final_for_answer = h3_output[i][0];
            final_answer = i+1;
        }
    }

    cout<<"Final Class Label (1 Based Indexing): "<<final_answer<<endl;
    disp(h3_output);
}