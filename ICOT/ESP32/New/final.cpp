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
#include "mlp_model_weights.cc"

#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>

#include "scaler.cc"

namespace fs = std::filesystem;

// #include <vector>
// #include <cmath>

// static inline float sinc(float x)
// {
//     if (fabs(x) < 1e-8f) return 1.0f;
//     return sinf(M_PI * x) / (M_PI * x);
// }

// static inline float kaiser(float x, float beta)
// {
//     float denom = std::cyl_bessel_i(0, beta);
//     float val = std::cyl_bessel_i(0, beta * sqrtf(1 - x * x));
//     return val / denom;
// }

// std::vector<float> librosa_resample(
//     const std::vector<float>& input,
//     int input_sr,
//     int target_sr)
// {
//     float ratio = (float)target_sr / input_sr;

//     int output_size = (int)(input.size() * ratio);

//     std::vector<float> output(output_size);

//     const int filter_width = 32;
//     const float beta = 14.769656459379492f;

//     for (int i = 0; i < output_size; i++)
//     {
//         float src_pos = i / ratio;
//         int center = (int)src_pos;

//         float sum = 0.0f;
//         float norm = 0.0f;

//         for (int k = -filter_width; k <= filter_width; k++)
//         {
//             int idx = center + k;

//             if (idx < 0 || idx >= input.size())
//                 continue;

//             float x = src_pos - idx;

//             float w = sinc(x) *
//                       kaiser(x / filter_width, beta);

//             sum += input[idx] * w;
//             norm += w;
//         }

//         output[i] = sum / norm;
//     }

//     return output;
// }

// std::vector<float> load_file(const std::string& path)
// {
//     std::ifstream file(path);
//     std::vector<float> values;
//     float v;

//     while(file >> v)
//         values.push_back(v);

//     return values;
// }

// constexpr size_t compute_final_size(size_t TOTAL_SAMPLES,size_t max_level,size_t wavelet_size){
//     size_t temp_final = 0;
//     size_t temp_total = TOTAL_SAMPLES;
//     size_t temp_level = max_level;
//     size_t temp_D = 0;
//     while(temp_level!=0){
//         temp_final = 2*((temp_total+wavelet_size-1)/2) + temp_D;
//         temp_D = temp_D + (temp_total+wavelet_size-1)/2;
//         temp_total = (temp_total+wavelet_size-1)/2;
//         temp_level--;
//     }
//     return temp_final;
// }

// struct WavHeader
// {
//     char riff[4];
//     uint32_t chunkSize;
//     char wave[4];

//     char fmt[4];
//     uint32_t subchunk1Size;
//     uint16_t audioFormat;
//     uint16_t numChannels;
//     uint32_t sampleRate;
//     uint32_t byteRate;
//     uint16_t blockAlign;
//     uint16_t bitsPerSample;

//     char data[4];
//     uint32_t dataSize;
// };

int main(){
    const int TARGET_SIZE = 32000;
    float audio[TARGET_SIZE] = {0.0f};

    std::ifstream file("output.wav", std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file\n";
        return 1;
    }

    // Skip WAV header (44 bytes)
    file.seekg(44, std::ios::beg);

    int16_t sample;
    int count = 0;

    while (file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t)) &&
           count < TARGET_SIZE)
    {
        // Convert PCM16 -> float
        audio[count] = (float)sample/32767.0f;
        count++;
    }

    file.close();

    std::cout << "Loaded " << count << " samples\n";

    cout<<"Hello World!"<<endl;
    static constexpr size_t FINAL_SIZE = 32038; //compute_final_size(32000,6,8);
    static constexpr size_t TOTAL_SAMPLES = 32000;
    static constexpr size_t sr = 16000;
    static constexpr size_t FFT_SIZE = 1024;
    static constexpr size_t HOP_SIZE = 256;
    static constexpr size_t NUM_FRAMES = 1 + (TOTAL_SAMPLES) / HOP_SIZE;
    static constexpr size_t NUM_BINS = FFT_SIZE/2 + 1;
    static constexpr size_t FINAL = FINAL_SIZE;// max(max((NUM_FRAMES+1)*NUM_BINS,TOTAL_SAMPLES),FINAL_SIZE);
    Matrix<float,FINAL,1> signal; // Change this so it will work for both dwt and stft.
    for(size_t i=0;i<FINAL;i++){
        signal[i][0] = 0;
    }
    for(size_t i=0;i<TOTAL_SAMPLES;i++){
        signal[i][0] = audio[i];
    }
    cout<<"RMS: "<<RMS(signal)<<endl<<endl;

    apply_pre_emphasis(signal,static_cast<float>(0.97));
    apply_bandpass_filter(signal);
    cout<<"Before: "<<RMS(signal)<<endl<<endl;
    normalize_rms(signal);
    // for(size_t i=0;i<25;i++){
    //     cout<<signal[i][0]<<endl;
    // }
    cout<<endl;


    Matrix<float,NUM_FRAMES*NUM_BINS,1>stft_output_actual;
    stft<float,TOTAL_SAMPLES,FFT_SIZE,HOP_SIZE,FINAL_SIZE> stft_object;
    stft_object.compute(signal,stft_output_actual);
    
    stft_output_actual.reset_shape();
    log_thingy(stft_output_actual);
    stft_output_actual.reshape(NUM_FRAMES,NUM_BINS);
    
    zoom_stft<float,NUM_FRAMES*NUM_BINS,NUM_FRAMES,NUM_BINS,100,128> zoom_stft_object;
    Matrix<float,100*128,1> stft_output = zoom_stft_object.zoom(stft_output_actual);
    
    stft_output.reset_shape();
    float mean_stft = mean(stft_output);
    float std_stft = standard_deviation(stft_output,mean_stft);
    cout<<"Mean "<<mean_stft<<endl;
    cout<<"STD: "<<std_stft<<endl;
    normalize_mean_std(stft_output,mean_stft,std_stft);
    for(size_t i=0;i<25;i++) cout<<stft_output[i][0]<<endl;
    cout<<endl;
    
    nan_inf_values(stft_output);
    clip(stft_output,static_cast<float>(-100),static_cast<float>(100));

    static constexpr size_t N = 100*128;
    static constexpr size_t r = 100;
    static constexpr size_t c = 128;

    Matrix<float,100*128,1> stft_positive;

    for(size_t i=0;i<N;i++){
        stft_positive[i][0] = abs_error(stft_output[i][0]);
    }

    Matrix<float,r,1> buffer1;
    Matrix<float,r,1> buffer2;
    Matrix<float,35,1> features_stft;
    Matrix<float,2,1> temp;

    stft_output.reset_shape();
    features_stft[0][0] = mean(stft_output);
    features_stft[1][0] = standard_deviation(stft_output,features_stft[0][0]);
    features_stft[2][0] = max_value(stft_output);
    features_stft[3][0] = min_value(stft_output);
    features_stft[4][0] = percentile(stft_output,features_stft[3][0],features_stft[2][0],static_cast<size_t>(0.25*(NUM_FRAMES*NUM_BINS)));
    features_stft[5][0] = percentile(stft_output,features_stft[3][0],features_stft[2][0],static_cast<size_t>(0.75*NUM_FRAMES*NUM_BINS));

    spectral_centroid<float,N,r,c>spectral_centroid_object;
    temp = spectral_centroid_object.compute(stft_positive,buffer1);
    features_stft[6][0] = temp[0][0];
    features_stft[7][0] = temp[1][0];

    spectral_bandwidth<float,N,r,c>spectral_bandwidth_object;
    temp = spectral_bandwidth_object.compute(stft_positive,buffer1,buffer2);
    features_stft[8][0] = temp[0][0];
    features_stft[9][0] = temp[1][0];

    spectral_rolloff<float,N,r,c>spectral_rolloff_object;
    temp = spectral_rolloff_object.compute(stft_positive,static_cast<float>(0.85),buffer1);
    features_stft[10][0] = temp[0][0];
    features_stft[11][0] = temp[1][0];

    spectral_flatness<float,N,r,c>spectral_flatness_object;
    temp = spectral_flatness_object.compute(stft_positive,buffer1);
    features_stft[12][0] = temp[0][0];
    features_stft[13][0] = temp[1][0];

    spectral_contrast<float,N,r,c>spectral_contrast_object;
    Matrix<float,4,1> constrast = spectral_contrast_object.compute(stft_positive,buffer1);
    features_stft[14][0] = constrast[0][0];
    features_stft[15][0] = constrast[1][0];
    features_stft[16][0] = constrast[2][0];
    features_stft[17][0] = constrast[3][0];

    features_stft[18][0] = zcr(stft_output,r,c);

    frequency_band_energies<float,N,r,c>frequency_band_energies_object;
    Matrix<float,5,1> fb_energies = frequency_band_energies_object.compute(stft_positive);
    features_stft[19][0] = fb_energies[0][0];
    features_stft[20][0] = fb_energies[1][0];
    features_stft[21][0] = fb_energies[2][0];
    features_stft[22][0] = fb_energies[3][0];
    features_stft[23][0] = fb_energies[4][0];

    temporal_features<float,N,r,c>temporal_features_object;
    Matrix<float,r-1,1> onset_strength;
    Matrix<float,5,1> temporal = temporal_features_object.compute(stft_output,buffer1,onset_strength);
    features_stft[24][0] = temporal[0][0];
    features_stft[25][0] = temporal[1][0];
    features_stft[26][0] = temporal[2][0];
    features_stft[27][0] = temporal[3][0];
    features_stft[28][0] = temporal[4][0];

    features_stft[29][0] = skewness(stft_output,features_stft[0][0],features_stft[1][0]);
    features_stft[30][0] = kurtosis(stft_output,features_stft[0][0],features_stft[1][0]);

    MFCC<float,N,r,c>MFCC_object;
    temp = MFCC_object.compute(stft_positive,sr);
    features_stft[31][0] = temp[0][0];
    features_stft[32][0] = temp[1][0];

    spectral_entropy_feature<float,N,r,c>spectral_entropy_feature_object;
    features_stft[33][0] = spectral_entropy_feature_object.compute(stft_positive);

    spectral_crest_factor<float,N>spectral_crest_factor_object;
    features_stft[34][0] = spectral_crest_factor_object.compute(stft_positive);
    

    nan_inf_values(features_stft);
    clip(features_stft,static_cast<float>(-1000),static_cast<float>(1000));

    disp(features_stft);

    dwt<float,TOTAL_SAMPLES,6> dwt_object;
    dwt_object.compute(signal);
    signal.reset_shape();
    for(size_t i=0;i<FINAL;i++){
        signal[i][0] = abs_error(signal[i][0]);
    }
    log_thingy(signal);
    signal.reset_shape();
    zoom_dwt<float,FINAL,7,16003,128,100> zoom_dwt_object;
    Matrix<float,100*128,1> dwt_output = zoom_dwt_object.zoom(signal);
    dwt_output.reset_shape();
    // dwt_output.reset_shape();
    // float mean_dwt = mean(dwt_output);
    // float std_dwt = standard_deviation(dwt_output,mean_dwt);

    // normalize_mean_std(dwt_output,mean_dwt,std_dwt);
    
    nan_inf_values(dwt_output);
    clip(dwt_output,static_cast<float>(-100),static_cast<float>(100));

    // static constexpr size_t N = 100*128;

    Matrix<float,100*128,1> dwt_positive;

    for(size_t i=0;i<N;i++){
        dwt_positive[i][0] = abs_error(dwt_output[i][0]);
    }

    for(size_t i=0;i<r;i++) buffer1[i][0] = 0;
    for(size_t i=0;i<r;i++) buffer2[i][0] = 0;

    Matrix<float,35,1> features_dwt;
    // Matrix<float,2,1> temp;

    dwt_output.reset_shape();
    features_dwt[0][0] = mean(dwt_output);
    features_dwt[1][0] = standard_deviation(dwt_output,features_dwt[0][0]);
    features_dwt[2][0] = max_value(dwt_output);
    features_dwt[3][0] = min_value(dwt_output);
    features_dwt[4][0] = percentile(dwt_output,features_dwt[3][0],features_dwt[2][0],static_cast<size_t>(0.25*FINAL));
    features_dwt[5][0] = percentile(dwt_output,features_dwt[3][0],features_dwt[2][0],static_cast<size_t>(0.75*FINAL));

    // spectral_centroid<float,N,r,c>spectral_centroid_object;
    temp = spectral_centroid_object.compute(dwt_positive,buffer1);
    features_dwt[6][0] = temp[0][0];
    features_dwt[7][0] = temp[1][0];

    // spectral_bandwidth<float,N,r,c>spectral_bandwidth_object;
    temp = spectral_bandwidth_object.compute(dwt_positive,buffer1,buffer2);
    features_dwt[8][0] = temp[0][0];
    features_dwt[9][0] = temp[1][0];

    // spectral_rolloff<float,N,r,c>spectral_rolloff_object;
    temp = spectral_rolloff_object.compute(dwt_positive,static_cast<float>(0.85),buffer1);
    features_dwt[10][0] = temp[0][0];
    features_dwt[11][0] = temp[1][0];

    // spectral_flatness<float,N,r,c>spectral_flatness_object;
    temp = spectral_flatness_object.compute(dwt_positive,buffer1);
    features_dwt[12][0] = temp[0][0];
    features_dwt[13][0] = temp[1][0];

    // spectral_contrast<float,N,r,c>spectral_contrast_object;
    // Matrix<float,4,1> constrast = spectral_contrast_object.compute(dwt_positive);
    constrast = spectral_contrast_object.compute(dwt_positive,buffer1);
    features_dwt[14][0] = constrast[0][0];
    features_dwt[15][0] = constrast[1][0];
    features_dwt[16][0] = constrast[2][0];
    features_dwt[17][0] = constrast[3][0];

    features_dwt[18][0] = zcr(dwt_output,r,c);

    // frequency_band_energies<float,N,r,c>frequency_band_energies_object;
    // Matrix<float,5,1> fb_energies = frequency_band_energies_object.compute(dwt_positive);
    fb_energies = frequency_band_energies_object.compute(dwt_positive);
    features_dwt[19][0] = fb_energies[0][0];
    features_dwt[20][0] = fb_energies[1][0];
    features_dwt[21][0] = fb_energies[2][0];
    features_dwt[22][0] = fb_energies[3][0];
    features_dwt[23][0] = fb_energies[4][0];

    // temporal_features<float,N,r,c>temporal_features_object;
    // Matrix<float,5,1> temporal = temporal_features_object.compute(dwt_output);
    temporal = temporal_features_object.compute(dwt_output,buffer1,onset_strength);
    features_dwt[24][0] = temporal[0][0];
    features_dwt[25][0] = temporal[1][0];
    features_dwt[26][0] = temporal[2][0];
    features_dwt[27][0] = temporal[3][0];
    features_dwt[28][0] = temporal[4][0];

    features_dwt[29][0] = skewness(dwt_output,features_dwt[0][0],features_dwt[1][0]);
    features_dwt[30][0] = kurtosis(dwt_output,features_dwt[0][0],features_dwt[1][0]);

    // MFCC<float,N,r,c>MFCC_object;
    temp = MFCC_object.compute(dwt_positive,sr);
    features_dwt[31][0] = temp[0][0];
    features_dwt[32][0] = temp[1][0];

    // spectral_entropy_feature<float,N,r,c>spectral_entropy_feature_object;
    features_dwt[33][0] = spectral_entropy_feature_object.compute(dwt_positive);

    // spectral_crest_factor<float,N>spectral_crest_factor_object;
    features_dwt[34][0] = spectral_crest_factor_object.compute(dwt_positive);
    

    nan_inf_values(features_dwt);
    clip(features_dwt,static_cast<float>(-1000),static_cast<float>(1000));

    disp(features_dwt);

    Matrix<float,70,1> final_features;
    for(size_t i=0;i<35;i++){
        final_features[i][0] = features_stft[i][0];
        final_features[i+35][0] = features_dwt[i][0];
    }
    
    // disp(features_stft);
    // disp(features_dwt);
    // float actual_output[70] = {1.907348590179936e-08, 1.0, 3.6468758583068848, -0.47718873620033264, -0.47718873620033264, -0.4727904200553894, 43.76631164550781, 2.7134110927581787, 39.1536750793457, 2.0439605712890625, 96.69999694824219, 6.738694190979004, 0.7698923349380493, 0.06799175590276718, 3.022700786590576, 0.0163809172809124, 0.007822866551578045, 0.002873064251616597, 0.13472576439380646, 4.2168707847595215, 0.21080376207828522, 0.22073788940906525, 0.22145256400108337, 0.22333475947380066, 127.99979400634766, 5.851312637329102, 155.2400360107422, 1.7282582521438599, 2.5295968055725098, 1.8905243873596191, 1.939000129699707, 0.9945995211601257, 2.8720388412475586, 9.158288955688477, 4.9376301765441895, -73.89167022705078, 13.608901977539062, -4.424343109130859, -80.0, -80.0, -77.78500366210938, 62.453155517578125, 3.318866491317749, 37.19174575805664, 1.3624324798583984, 108.20999908447266, 2.479092597961426, 0.985016942024231, 0.02920292690396309, 1.7041305303573608, 4.649719715118408, 10.271296501159668, 13.918478012084961, 0.20663265883922577, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 2.4793474674224854, 5.397080421447754, -66.66077423095703, 226.13731384277344, 9.436251640319824, 1.082666039466858};
    // float actual_output[70] = {0.0, 1.0, 3.4023990631103516, -0.48269134759902954, -0.48269134759902954, -0.4597267508506775, 44.24533462524414, 2.868252754211426, 39.219783782958984, 1.68180251121521, 97.0199966430664, 5.892333984375, 0.7789808511734009, 0.05270480364561081, 2.9090137481689453, 0.01578073762357235, 0.006483786273747683, 0.0018602487398311496, 0.13719706237316132, 4.169139862060547, 0.23212985694408417, 0.2343413531780243, 0.2293490171432495, 0.2277141511440277, 128.00025939941406, 12.463654518127441, 239.34259033203125, 1.8701233863830566, 2.9850406646728516, 1.908902883529663, 2.0668840408325195, 0.9905628561973572, 2.8639121055603027, 9.163562774658203, 4.6033525466918945, -74.15657043457031, 13.226701736450195, -1.5143814086914062, -80.0, -80.0, -78.70388793945312, 62.63661193847656, 3.632392644882202, 37.16407012939453, 1.1551730632781982, 108.27999877929688, 2.391986608505249, 0.9863796830177307, 0.026776840910315514, 1.386080265045166, 4.232049942016602, 10.153007507324219, 13.068385124206543, 0.2014508992433548, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 2.5187976360321045, 5.646012306213379, -66.64530944824219, 226.77175903320312, 9.437643051147461, 1.078798532485962};
    // disp(final_features);
    // for(size_t i=0;i<70;i++) final_features[i][0] = actual_output[i];
    clip(final_features,static_cast<float>(-1000),static_cast<float>(1000));
    
    for(size_t i=0;i<70;i++){
        final_features[i][0] = (final_features[i][0]-scaler::scaler_mean[i])/scaler::scaler_std[i];
    }
    Matrix<float,70,64> hidden_1_w;
    hidden_1_w.reshape(70*64,1);
    for(size_t i=0;i<70*64;i++){
        hidden_1_w[i][0] = mlp_weights::hidden_1_weights[i];
    }
    hidden_1_w.reset_shape();

    Matrix<float,64,32> hidden_2_w;
    hidden_2_w.reshape(64*32,1);
    for(size_t i=0;i<64*32;i++){
        hidden_2_w[i][0] = mlp_weights::hidden_2_weights[i];
    }
    hidden_2_w.reset_shape();

    Matrix<float,32,6> hidden_3_w;
    hidden_3_w.reshape(32*6,1);
    for(size_t i=0;i<32*6;i++){
        hidden_3_w[i][0] = mlp_weights::student_output_weights[i];
    }
    hidden_3_w.reset_shape();


    Matrix<float,64,1> hidden_1_b;
    for(size_t i=0;i<64;i++){
        hidden_1_b[i][0] = mlp_weights::hidden_1_bias[i];
    }

    Matrix<float,32,1> hidden_2_b;
    for(size_t i=0;i<32;i++){
        hidden_2_b[i][0] = mlp_weights::hidden_2_bias[i];
    }

    Matrix<float,6,1> hidden_3_b;
    for(size_t i=0;i<6;i++){
        hidden_3_b[i][0] = mlp_weights::student_output_bias[i];
    }

    Dense<float,64,70> hidden_1;
    Dense<float,32,64> hidden_2;
    Dense<float,6,32> hidden_3;

    hidden_1.setWeight(hidden_1_w);
    hidden_2.setWeight(hidden_2_w);
    hidden_3.setWeight(hidden_3_w);

    hidden_1.setBias(hidden_1_b);
    hidden_2.setBias(hidden_2_b);
    hidden_3.setBias(hidden_3_b);

    auto h1_output = hidden_1.forward(final_features);
    relu(h1_output);
    auto h2_output = hidden_2.forward(h1_output);
    relu(h2_output);
    auto h3_output = hidden_3.forward(h2_output);
    disp(h3_output);
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