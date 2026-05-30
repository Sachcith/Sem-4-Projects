#ifndef feature_H
#define feature_H
#include <cstddef>
#include <cmath>
#include "Matrix.hpp"

template <class T,std::size_t N>
class feature{
    private:
        Matrix<T,35,1> features;
    public:


};

// Mean Function
template <class T,std::size_t N>
T mean(Matrix<T,N,1> &input){
    if(N==0) return 0;
    T total = 0;
    for(std::size_t i=0;i<N;i++){
        total = total + input[i][0];
    }
    return total/N;
}

// Standard Deviation Function
template <class T,std::size_t N>
T standard_deviation(Matrix<T,N,1> &input,T input_mean){
    T total = 0;
    for(std::size_t i=0;i<N;i++){
        total = total + (input[i][0]-input_mean)*(input[i][0]-input_mean);
    }
    // return std::sqrt(total/(N-1));
    return std::sqrt(total/(N));
}

// Maximum Value
template <class T,std::size_t N>
T max_value(Matrix<T,N,1> &input){
    T temp_max = input[0][0];
    for(std::size_t i=1;i<N;i++){
        if(temp_max<input[i][0]){
            temp_max = input[i][0];
        }
    }
    return temp_max;
}

// Minimum Value
template <class T,std::size_t N>
T min_value(Matrix<T,N,1> &input){
    T temp_min = input[0][0];
    for(std::size_t i=1;i<N;i++){
        if(temp_min>input[i][0]){
            temp_min = input[i][0];
        }
    }
    return temp_min;
}

// Value Space Binary Search on Unsorted Array
// Finds kth smallest element on an unsorted array with time complexity O(nlogn) and space complexity O(1).
// Here low means min value and high means max value in the total array.
template <class T,std::size_t N>
T percentile(Matrix<T,N,1> &input,T low,T high,std::size_t k){
    k--;
    T mid = (low+high)/2;
    std::size_t max_iterations = 25;
    while(max_iterations!=0 && low<=high){
        mid = (low+high)/2;
        std::size_t count = 0;
        for(std::size_t i=0;i<N;i++){
            if(input[i][0]<=mid) count++;
        }
        if(count>k){
            high = mid;
        }
        else{
            low = mid;
        }
        max_iterations--;
    }
    return low;
}

// Skewness Function
// Skewness = N/((N−1)*(N−2)) * ∑((x[i]​−μ)/s)^3
// ​μ = mean
// s = standard deviation
template <class T,std::size_t N>
T skewness(Matrix<T,N,1> &input,T mean_value,T std_dev){
    T total = 0;
    T temp = 0;
    for(std::size_t i=0;i<N;i++){
        temp = (input[i][0]-mean_value)/std_dev;
        total = total + temp*temp*temp;
    }
    return total/N;
}

// Kurtosis Function
// Kurtosis = (N*(N+1))/((N-1)*(N-2)*(N-3)) * ∑((x[i]-μ)/s)^4 - (3*(N-1)^2)/((N-2)*(N-3))
// ​μ = mean
// s = standard deviation
template <class T,std::size_t N>
T kurtosis(Matrix<T,N,1> &input,T mean_value,T std_dev){
    T total = 0;
    T temp = 0;
    for(std::size_t i=0;i<N;i++){
        temp = (input[i][0]-mean_value)/std_dev;
        total = total + temp*temp*temp*temp;
    }
    return total/N - 3;
}

// Zero Crossing Rate Function (ZCR)
template <class T,std::size_t N>
T zcr(Matrix<T,N,1> &input,std::size_t NUM_FRAMES,std::size_t NUM_BINS){
    input.reshape(NUM_FRAMES,NUM_BINS);
    T count = 0;
    T temp1;
    T temp2;
    for(std::size_t i=0;i<NUM_FRAMES;i++){
        for(std::size_t j=2;j<NUM_BINS;j++){
            T temp1 = input[i][j-1] - input[i][j-2];
            T temp2 = input[i][j] - input[i][j-1];
            int s1 = (temp1 > 0) - (temp1 < 0);
            int s2 = (temp2 > 0) - (temp2 < 0);
            count += std::abs((T)s2 - (T)s1) / (T)2;
        }
    }
    input.reset_shape();
    return count/((NUM_FRAMES)*(NUM_BINS-2));
}

// Root Mean Square Energy Function (RMS)
template <class T,std::size_t N>
T RMS(const Matrix<T,N,1> &input){
    T total = 0;
    for(std::size_t i=0;i<N;i++){
        total = total + input[i][0]*input[i][0];
    }
    return std::sqrt(total/static_cast<T>(N));
}

// Spectral Centroid
// Centroid Calculator thingy
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class spectral_centroid{
    public:
        Matrix<T,2,1> compute(Matrix<T,N,1> &input,Matrix<T,NUM_FRAMES,1> &output){
            input.reshape(NUM_FRAMES,NUM_BINS);
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T mag_total = 0;
                output[i][0] = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
                    T fi = j;
                    output[i][0] = output[i][0] + input[i][j]*fi;
                    mag_total = mag_total + input[i][j];
                }
                if(mag_total==0) output[i][0] = 1e-10;
                else output[i][0] = output[i][0]/mag_total;
            }

            input.reset_shape();
            Matrix<T,2,1> actual_output;
            actual_output[0][0] = mean(output);
            actual_output[1][0] = standard_deviation(output,actual_output[0][0]);
            return actual_output;
        }
};

// Spectral Bandwidth Function
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class spectral_bandwidth{
    public:
        Matrix<T,2,1> compute(Matrix<T,N,1> &input, Matrix<T,NUM_FRAMES,1> &centroid, Matrix<T,NUM_FRAMES,1> &bandwidth){
            input.reshape(NUM_FRAMES,NUM_BINS);
            std::size_t FFT_SIZE = (NUM_BINS-1)*2;
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T mag_total = 0;
                centroid[i][0] = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
                    centroid[i][0] = centroid[i][0] + input[i][j]*j;
                    mag_total = mag_total + input[i][j];
                }
                if(mag_total<=1e-10) centroid[i][0] = 0;
                else centroid[i][0] = centroid[i][0]/mag_total;
            }

            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T mag_sum = 0;
                T square = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
                    mag_sum = mag_sum + input[i][j];
                    square = square + (j-centroid[i][0])*(j-centroid[i][0])*input[i][j];
                }
                if(mag_sum<=1e-10 || square<=0) bandwidth[i][0] = 0;
                else bandwidth[i][0] = std::sqrt(square/mag_sum);
            }
            input.reset_shape();
            Matrix<T,2,1> output;
            output[0][0] = mean(bandwidth);
            output[1][0] = standard_deviation(bandwidth,output[0][0]);
            return output;
        }
};
// Centroid Function for helping Spectral Bandwidth Function
/*
template <class T, std::size_t N, std::size_t NUM_FRAMES>
Matrix<T,NUM_FRAMES,1> centroid_spectral(Matrix<T,N,1> &input,std::size_t sample_rate, std::size_t NUM_FRAMES, std::size_t NUM_BINS){
    Matrix<T,NUM_FRAMES,1> output;
    std::size_t FFT_SIZE = (NUM_BINS-1)*2;
    for(std::size_t i=0;i<NUM_FRAMES;i++){
        T mag_total = 0;
        for(std::size_t j=0;j<NUM_BINS;j++){
            output[i][0] = output[i][0] + input[i][j]*j;
            mag_total = mag_total + input[i][j];
        }
        if(mag_total==0) output[i][0] = 0;
        else output[i][0] = output[i][0]/mag_total;
    }
    return output;
}
*/

// Spectral Rolloff Function
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class spectral_rolloff{
    public:
        Matrix<T,2,1> compute(Matrix<T,N,1> &input,T percentile, Matrix<T,NUM_FRAMES,1> &rolloff){
            input.reshape(NUM_FRAMES,NUM_BINS);
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T mag_sum = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
                    mag_sum = mag_sum + input[i][j];
                }
                
                if(mag_sum==0) mag_sum = 1e-10;

                T thresh = percentile*mag_sum;
                T prefixsum = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
                    prefixsum = prefixsum + input[i][j];
                    if(prefixsum>=thresh){
                        rolloff[i][0] = j;
                        break;
                    }
                }
            }
            input.reset_shape();
            Matrix<T,2,1> output;
            output[0][0] = mean(rolloff);
            output[0][1] = standard_deviation(rolloff,output[0][0]);
            return output;
        }
};

// Spectral Flatness Function
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class spectral_flatness{

    public:
        Matrix<T,2,1> compute(Matrix<T,N,1> &input,Matrix<T,NUM_FRAMES,1> &flatness){
            input.reshape(NUM_FRAMES,NUM_BINS);
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T mag_sum = 0;
                T log_sum = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
                    T temp = std::abs(input[i][j]);
                    if(temp<1e-10) temp = 1e-10;
                    mag_sum = mag_sum + temp;
                    log_sum = log_sum + std::log(temp);
                }
                if(NUM_BINS>0) mag_sum = mag_sum/NUM_BINS;
                if(mag_sum<=1e-10) mag_sum = 1e-10;
                else flatness[i][0] = std::exp(log_sum/NUM_BINS)/mag_sum;
            }
            input.reset_shape();
            Matrix<T,2,1> output;
            output[0][0] = mean(flatness);
            output[1][0] = standard_deviation(flatness,output[0][0]);
            return output;
        }
};

// Spectral Contrast Function
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class spectral_contrast{
    private:
            Matrix<T,4,1> contrast;
    public:
        Matrix<T,4,1> compute(Matrix<T,N,1> &input, Matrix<T,NUM_FRAMES,1> &temp){
            input.reshape(NUM_FRAMES,NUM_BINS);

            std::size_t n_bands = 4;
            T band_size = NUM_BINS/n_bands;
            for(std::size_t k=0;k<n_bands;k++){
                for(std::size_t i=0;i<NUM_FRAMES;i++){
                    T temp_max = input[i][static_cast<std::size_t>(k*band_size)];
                    T temp_min = input[i][static_cast<std::size_t>(k*band_size)];

                    std::size_t end = (k+1)*band_size;
                    if(k==n_bands-1) end = NUM_BINS;

                    for(std::size_t j=k*band_size;j<end;j++){
                        temp_max = max(temp_max,input[i][j]);
                        temp_min = min(temp_min,input[i][j]);
                    }
                    temp[i][0] = temp_max - temp_min;
                }
                contrast[k][0] = mean(temp);
            }
            input.reset_shape();
            return contrast;
        }
};

// Frequency Band Energy Function
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class frequency_band_energies{
    public:
        Matrix<T,5,1> compute(Matrix<T,N,1> &input){
            input.reshape(NUM_FRAMES,NUM_BINS);
            std::size_t n_bands = 5;
            std::size_t band_size = NUM_BINS/n_bands;
            Matrix<T,5,1> FBE;
            for(std::size_t k=0;k<n_bands;k++){
                std::size_t end = (k+1)*band_size;
                if(k==n_bands-1) end = NUM_BINS;
                T temp = 0;
                T count = 0;
                for(std::size_t i=0;i<NUM_FRAMES;i++){
                    for(std::size_t j=k*band_size;j<end;j++){
                        temp = temp + input[i][j]*input[i][j];
                        count++;
                    }
                }
                FBE[k][0] = temp/count;
            }
            input.reset_shape();
            return FBE;
        }
};

// Temporal Features Function
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class temporal_features{
    public:
        Matrix<T,5,1> compute(Matrix<T,N,1> &input, Matrix<T,NUM_FRAMES,1> frame_energy, Matrix<T,NUM_FRAMES-1,1> onset_strength){
            input.reshape(NUM_FRAMES,NUM_BINS);
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T energy = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
                    energy = energy + input[i][j]*input[i][j];
                }
                frame_energy[i][0] = energy;
            }
            for(std::size_t i=1;i<NUM_FRAMES;i++){
                onset_strength[i-1][0] = max(static_cast<T>(0),frame_energy[i][0]-frame_energy[i-1][0]);
            }
            input.reset_shape();
            Matrix<T,5,1> temporal;
            temporal[0][0] = mean(frame_energy);
            temporal[1][0] = standard_deviation(frame_energy,temporal[0][0]);
            temporal[2][0] = max_value(frame_energy);
            if(NUM_FRAMES-1>0){
                temporal[3][0] = mean(onset_strength);
                temporal[4][0] = standard_deviation(onset_strength,temporal[3][0]);
            }
            else{
                temporal[3][0] = 0;
                temporal[4][0] = 0;
            }
            return temporal;
        }
};

// Spectral Entropy Feature Function
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class spectral_entropy_feature{
    public:
        T compute(Matrix<T,N,1> &input){
            input.reshape(NUM_FRAMES,NUM_BINS);
            T total_sum = 0;
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                for(std::size_t j=0;j<NUM_BINS;j++){
                    total_sum = total_sum + input[i][j];
                }
            }
            T entropy = 0;
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                for(std::size_t j=0;j<NUM_BINS;j++){
                    if(input[i][j]/(total_sum + 1e-10) > 0) entropy = entropy - (input[i][j]/(total_sum + 1e-10))*std::log(input[i][j]/(total_sum + 1e-10));
                }
            }
            input.reset_shape();
            return entropy;
        }
};

// Spectral Crest Factor Function
template <class T, std::size_t N>
class spectral_crest_factor{
    public:
        T compute(Matrix<T,N,1> &input){
            input.reset_shape();
            T m = mean(input);
            return max_value(input)/(m+1e-10);
        }
};

// MFCC Class
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class MFCC{
    public:
        // Mel-Frequency Cepstral Coefficients Function (MFCC)
        Matrix<T,2,1> compute(Matrix<T,N,1> &input, std::size_t sr){ // sr = sample rate
            Matrix<T,13*NUM_FRAMES,1> mfcc;
            mfcc.reshape(13,NUM_FRAMES);
            // for(std::size_t i=0;i<N;i++) input[i][0] = std::log10(std::abs(input[i][0])+1e-10);
            input.reshape(NUM_FRAMES,NUM_BINS);
            // for(std::size_t t = 0; t < NUM_FRAMES; t++){
            //     for(std::size_t c = 0; c < 13; c++){
            //         mfcc[c][t] = 0;
            //         for(std::size_t m = 0; m < NUM_BINS; m++){
            //             mfcc[c][t] += input[t][m] *
            //                         std::cos(M_PI * c * (m + 0.5) / NUM_BINS);}
            //     }
            // }
            for(size_t c = 0; c < 13; c++){
                for(size_t t = 0; t < NUM_FRAMES; t++){
                    double sum = 0;
                    for(size_t m = 0; m < NUM_BINS; m++){
                        sum += input[t][m] *
                            std::cos(M_PI * (m + 0.5) * c / NUM_BINS);
                    }
                    if(c == 0) mfcc[c][t] = sum * std::sqrt(1.0 / NUM_BINS);
                    else mfcc[c][t] = sum * std::sqrt(2.0 / NUM_BINS);
                }
            }
            mfcc.reset_shape();
            input.reset_shape();
            
            Matrix<T,2,1> output;
            output[0][0] = mean(mfcc);
            output[1][0] = standard_deviation(mfcc,output[0][0]);
            return output;
        }
};

template <class T, std::size_t N>
void normalize_mean_std(Matrix<T,N,1> &input, T mean_value, T standard_dev){
    standard_dev = standard_dev + 1e-8;
    for(std::size_t i=0;i<N;i++){
        input[i][0] = (input[i][0] - mean_value)/standard_dev;
    }
}

template <class T, std::size_t N>
void clip(Matrix<T,N,1> &input, T min_val, T max_val){
    for(std::size_t i=0;i<N;i++){
        input[i][0] = min(max_val,max(min_val,input[i][0]));
    }
}

template <class T>
T abs_error(T &input){
    return std::abs(input) + 1e-10;
}

template <class T, std::size_t N>
void nan_inf_values(Matrix<T,N,1> &input){
    for(std::size_t i=0;i<N;i++){
        if(std::isnan(input[i][0])){
            input[i][0] = 0;
        }
        else if(std::isinf(input[i][0])){
            if(input[i][0]>0) input[i][0] = 1;
            else input[i][0] = -1;
        }
    }
}
#endif