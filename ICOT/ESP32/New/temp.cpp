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

#include <fstream>


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
    const int TARGET_SIZE = 32000;
    float audio[TARGET_SIZE] = {0.0f};

    std::ifstream file("input.wav", std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file.\n";
    }

    // Skip standard 44-byte WAV header
    file.seekg(44, std::ios::beg);

    int16_t sample;
    int count = 0;

    while (file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t)) && count < TARGET_SIZE) {
        // Convert 16-bit PCM to float (-1.0 to 1.0)
        audio[count] = static_cast<float>(sample) / 32768.0f;
        count++;
    }

    file.close();

    std::cout << "Read " << count << " samples into float array.\n";


    cout<<"Hello World!"<<endl;
    static constexpr size_t FINAL_SIZE = compute_final_size(32000,6,8);
    static constexpr size_t TOTAL_SAMPLES = 32000;
    static constexpr size_t sr = 16000;
    static constexpr size_t FFT_SIZE = 1024;
    static constexpr size_t HOP_SIZE = 512;
    static constexpr size_t NUM_FRAMES = (TOTAL_SAMPLES - FFT_SIZE) / HOP_SIZE;
    static constexpr size_t NUM_BINS = FFT_SIZE/2 + 1;
    static constexpr size_t FINAL = max(max((NUM_FRAMES+1)*NUM_BINS,TOTAL_SAMPLES),FINAL_SIZE);
    Matrix<float,FINAL,1> signal; // Change this so it will work for both dwt and dwt.
    for(size_t i=0;i<FINAL;i++){
        signal[i][0] = 1;
    }
    for(size_t i=0;i<TOTAL_SAMPLES;i++){
        signal[i][0] = audio[i];
    }


    apply_pre_emphasis(signal,static_cast<float>(0.97));
    //apply_bandpass_filter(signal);
    normalize_rms(signal);

    dwt<float,TOTAL_SAMPLES,6> dwt_object;
    dwt_object.compute(signal);
    log_thingy(signal);

    zoom_dwt<float,FINAL,7,16003,100,128> zoom_dwt_object;
    Matrix<float,100*128,1> dwt_output = zoom_dwt_object.zoom(signal);
    
    dwt_output.reset_shape();
    float mean_dwt = mean(dwt_output);
    float std_dwt = standard_deviation(dwt_output,mean_dwt);

    normalize_mean_std(dwt_output,mean_dwt,std_dwt);
    
    nan_inf_values(dwt_output);
    clip(dwt_output,static_cast<float>(-100),static_cast<float>(100));

    static constexpr size_t N = 100*128;

    Matrix<float,100*128,1> dwt_positive;

    for(size_t i=0;i<N;i++){
        dwt_positive[i][0] = abs_error(dwt_output[i][0]);
    }

    Matrix<float,35,1> features_dwt;
    Matrix<float,2,1> temp;

    dwt_output.reset_shape();
    features_dwt[0][0] = mean(dwt_output);
    features_dwt[1][0] = standard_deviation(dwt_output,features_dwt[0][0]);
    features_dwt[2][0] = max_value(dwt_output);
    features_dwt[3][0] = min_value(dwt_output);
    features_dwt[4][0] = percentile(dwt_output,features_dwt[3][0],features_dwt[2][0],static_cast<size_t>(0.25*TOTAL_SAMPLES));
    features_dwt[5][0] = percentile(dwt_output,features_dwt[3][0],features_dwt[2][0],static_cast<size_t>(0.75*TOTAL_SAMPLES));

    spectral_centroid<float,N,NUM_FRAMES,NUM_BINS>spectral_centroid_object;
    temp = spectral_centroid_object.compute(dwt_positive,sr);
    features_dwt[6][0] = temp[0][0];
    features_dwt[7][0] = temp[1][0];

    spectral_bandwidth<float,N,NUM_FRAMES,NUM_BINS>spectral_bandwidth_object;
    temp = spectral_bandwidth_object.compute(dwt_positive);
    features_dwt[8][0] = temp[0][0];
    features_dwt[9][0] = temp[1][0];

    spectral_rolloff<float,N,NUM_FRAMES,NUM_BINS>spectral_rolloff_object;
    temp = spectral_rolloff_object.compute(dwt_positive,static_cast<float>(0.85));
    features_dwt[10][0] = temp[0][0];
    features_dwt[11][0] = temp[1][0];

    spectral_flatness<float,N,NUM_FRAMES,NUM_BINS>spectral_flatness_object;
    temp = spectral_flatness_object.compute(dwt_positive);
    features_dwt[12][0] = temp[0][0];
    features_dwt[13][0] = temp[1][0];

    spectral_contrast<float,N,NUM_FRAMES,NUM_BINS>spectral_contrast_object;
    Matrix<float,4,1> constrast = spectral_contrast_object.compute(dwt_positive);
    features_dwt[14][0] = constrast[0][0];
    features_dwt[15][0] = constrast[1][0];
    features_dwt[16][0] = constrast[2][0];
    features_dwt[17][0] = constrast[3][0];

    features_dwt[18][0] = zcr(dwt_output);

    frequency_band_energies<float,N,NUM_FRAMES,NUM_BINS>frequency_band_energies_object;
    Matrix<float,5,1> fb_energies = frequency_band_energies_object.compute(dwt_positive);
    features_dwt[19][0] = fb_energies[0][0];
    features_dwt[20][0] = fb_energies[1][0];
    features_dwt[21][0] = fb_energies[2][0];
    features_dwt[22][0] = fb_energies[3][0];
    features_dwt[23][0] = fb_energies[4][0];

    temporal_features<float,N,NUM_FRAMES,NUM_BINS>temporal_features_object;
    Matrix<float,5,1> temporal = temporal_features_object.compute(dwt_output);
    features_dwt[24][0] = temporal[0][0];
    features_dwt[25][0] = temporal[1][0];
    features_dwt[26][0] = temporal[2][0];
    features_dwt[27][0] = temporal[3][0];
    features_dwt[28][0] = temporal[4][0];

    features_dwt[29][0] = skewness(dwt_output,features_dwt[0][0],features_dwt[1][0]);
    features_dwt[30][0] = kurtosis(dwt_output,features_dwt[0][0],features_dwt[1][0]);

    MFCC<float,N,NUM_FRAMES,NUM_BINS>MFCC_object;
    temp = MFCC_object.compute(dwt_positive,sr);
    features_dwt[31][0] = temp[0][0];
    features_dwt[32][0] = temp[1][0];

    spectral_entropy_feature<float,N,NUM_FRAMES,NUM_BINS>spectral_entropy_feature_object;
    features_dwt[33][0] = spectral_entropy_feature_object.compute(dwt_positive);

    spectral_crest_factor<float,N>spectral_crest_factor_object;
    features_dwt[34][0] = spectral_crest_factor_object.compute(dwt_positive);
    
    disp(features_dwt);

    // dwt_output.reshape(100,128);
    // disp(dwt_output);
    // dwt_output.reset_shape();


}