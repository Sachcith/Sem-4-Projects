#include <iostream>
#include <bits/stdc++.h>
using namespace std;

#include "Matrix.hpp"
#include "disp.hpp"
#include "Dense.hpp"
#include "activation.hpp"
#include "stft.hpp"
#include "dwt.hpp"
#include "feature.hpp"
#include "audio_cleaner.hpp"


constexpr size_t compute_final_size(size_t TOTAL_SAMPLES,size_t max_level,size_t wavelet_size){
    size_t temp_final = 0;
    size_t temp_total = TOTAL_SAMPLES;
    size_t temp_level = max_level;
    size_t temp_D = 0;
    while(temp_level!=0){
        temp_final = 2*((temp_total+wavelet_size-1)/2) + temp_D;
        temp_D = temp_D + (temp_total+wavelet_size-1)/2;
        temp_total = (temp_total+wavelet_size-1)/2;
        temp_level--;
    }
    return temp_final;
}


int main(){
    cout<<"Hello World!"<<endl;
    static constexpr size_t FINAL_SIZE = compute_final_size(32000,6,8);
    static constexpr size_t TOTAL_SAMPLES = 32000;
    static constexpr size_t sr = 16000;
    static constexpr size_t FFT_SIZE = 1024;
    static constexpr size_t HOP_SIZE = 512;
    static constexpr size_t NUM_FRAMES = (TOTAL_SAMPLES - FFT_SIZE) / HOP_SIZE;
    static constexpr size_t NUM_BINS = FFT_SIZE/2 + 1;
    static constexpr size_t FINAL = max((NUM_FRAMES+1)*NUM_BINS,TOTAL_SAMPLES);
    Matrix<float,FINAL,1> signal; // Change this so it will work for both dwt and stft.
    for(size_t i=0;i<FINAL;i++){
        signal[i][0] = 1;
    }


    apply_pre_emphasis(signal,static_cast<float>(0.97));
    //apply_bandpass_filter(signal);
    normalize_rms(signal);

    stft<float,TOTAL_SAMPLES,FFT_SIZE,HOP_SIZE> stft_object;
    stft_object.compute(signal);
    log_thingy(signal);

    zoom_stft<float,FINAL,NUM_FRAMES,NUM_BINS,100,128> zoom_stft_object;
    Matrix<float,100*128,1> stft_output = zoom_stft_object.zoom(signal);
    
    stft_output.reset_shape();
    float mean_stft = mean(stft_output);
    float std_stft = standard_deviation(stft_output,mean_stft);

    normalize_mean_std(stft_output,mean_stft,std_stft);
    
    nan_inf_values(stft_output);

    static constexpr size_t N = 100*128;

    Matrix<float,100*128,1> stft_positive = zoom_stft_object.zoom(signal);

    for(size_t i=0;i<N;i++){
        stft_positive[i][0] = abs_error(stft_output[i][0]);
    }

    Matrix<float,35,1> features_stft;
    Matrix<float,2,1> temp;

    stft_output.reset_shape();
    features_stft[0][0] = mean(stft_output);
    features_stft[1][0] = standard_deviation(stft_output,features_stft[0][0]);
    features_stft[2][0] = max_value(stft_output);
    features_stft[3][0] = min_value(stft_output);
    features_stft[4][0] = percentile(stft_output,features_stft[3][0],features_stft[2][0],static_cast<size_t>(0.25*TOTAL_SAMPLES));
    features_stft[5][0] = percentile(stft_output,features_stft[3][0],features_stft[2][0],static_cast<size_t>(0.75*TOTAL_SAMPLES));

    spectral_centroid<float,N,NUM_FRAMES,NUM_BINS>spectral_centroid_object;
    temp = spectral_centroid_object.compute(stft_positive,sr);
    features_stft[6][0] = temp[0][0];
    features_stft[7][0] = temp[1][0];

    spectral_bandwidth<float,N,NUM_FRAMES,NUM_BINS>spectral_bandwidth_object;
    temp = spectral_bandwidth_object.compute(stft_positive);
    features_stft[8][0] = temp[0][0];
    features_stft[9][0] = temp[1][0];

    spectral_rolloff<float,N,NUM_FRAMES,NUM_BINS>spectral_rolloff_object;
    temp = spectral_rolloff_object.compute(stft_positive,static_cast<float>(0.85));
    features_stft[10][0] = temp[0][0];
    features_stft[11][0] = temp[1][0];

    spectral_flatness<float,N,NUM_FRAMES,NUM_BINS>spectral_flatness_object;
    temp = spectral_flatness_object.compute(stft_positive);
    features_stft[12][0] = temp[0][0];
    features_stft[13][0] = temp[1][0];

    spectral_contrast<float,N,NUM_FRAMES,NUM_BINS>spectral_contrast_object;
    Matrix<float,4,1> constrast = spectral_contrast_object.compute(stft_positive);
    features_stft[14][0] = constrast[0][0];
    features_stft[15][0] = constrast[1][0];
    features_stft[16][0] = constrast[2][0];
    features_stft[17][0] = constrast[3][0];

    features_stft[18][0] = zcr(stft_output);

    frequency_band_energies<float,N,NUM_FRAMES,NUM_BINS>frequency_band_energies_object;
    Matrix<float,5,1> fb_energies = frequency_band_energies_object.compute(stft_positive);
    features_stft[19][0] = fb_energies[0][0];
    features_stft[20][0] = fb_energies[1][0];
    features_stft[21][0] = fb_energies[2][0];
    features_stft[22][0] = fb_energies[3][0];
    features_stft[23][0] = fb_energies[4][0];

    temporal_features<float,N,NUM_FRAMES,NUM_BINS>temporal_features_object;
    Matrix<float,5,1> temporal = temporal_features_object.compute(stft_output);
    features_stft[24][0] = temporal[0][0];
    features_stft[25][0] = temporal[1][0];
    features_stft[26][0] = temporal[2][0];
    features_stft[27][0] = temporal[3][0];
    features_stft[28][0] = temporal[4][0];

    features_stft[29][0] = skewness(stft_output,features_stft[0][0],features_stft[1][0]);
    features_stft[30][0] = kurtosis(stft_output,features_stft[0][0],features_stft[1][0]);

    MFCC<float,N,NUM_FRAMES,NUM_BINS>MFCC_object;
    temp = MFCC_object.compute(stft_output,sr);
    features_stft[31][0] = temp[0][0];
    features_stft[32][0] = temp[1][0];

    spectral_entropy_feature<float,N,NUM_FRAMES,NUM_BINS>spectral_entropy_feature_object;
    features_stft[33][0] = spectral_entropy_feature_object.compute(stft_positive);

    spectral_crest_factor<float,N>spectral_crest_factor_object;
    features_stft[34][0] = spectral_crest_factor_object.compute(stft_positive);
    
    disp(features_stft);

    // stft_output.reshape(100,128);
    // disp(stft_output);
    // stft_output.reset_shape();


}