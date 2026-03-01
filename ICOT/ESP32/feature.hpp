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
    T mid = (low+high)/2;
    // Tolarance = 1e-3;
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
    T s3 = std_dev*std_dev*std_dev;
    T temp = 0;
    for(std::size_t i=0;i<N;i++){
        temp = input[i][0]-mean_value;
        total = total + temp*temp*temp;
    }
    T c = N/((N-1)*(N-2));
    return c*total/s3;
}

// Kurtosis Function
// Kurtosis = (N*(N+1))/((N-1)*(N-2)*(N-3)) * ∑((x[i]-μ)/s)^4 - (3*(N-1)^2)/((N-2)*(N-3))
// ​μ = mean
// s = standard deviation
template <class T,std::size_t N>
T kurtosis(Matrix<T,N,1> &input,T mean_value,T std_dev){
    T total = 0;
    T s4 = std_dev*std_dev*std_dev*std_dev;
    T temp = 0;
    for(std::size_t i=0;i<N;i++){
        temp = input[i][0]-mean_value;
        total = total + temp*temp*temp*temp;
    }
    T c = (N*(N+1))/((N-1)*(N-2)*(N-3));
    T sub = (3*(N-1)*(N-1))/((N-2)*(N-3));
    return c*total/s4 - sub;
}

// Zero Crossing Rate Function (ZCR)
template <class T,std::size_t N>
T zcr(Matrix<T,N,1> &input){
    T count = 0;
    T temp1;
    T temp2;
    for(std::size_t i=2;i<N;i++){
        T temp1 = input[i-1] - input[i-2];
        T temp2 = input[i] - input[i-1];
        if(temp1*temp2<0){
            count++;
        }
    }
    return count;
}

// Root Mean Square Energy Function (RMS)
template <class T,std::size_t N>
T RMS(Matrix<T,N,1> &input){
    T total = 0;
    for(std::size_t i=0;i<N;i++){
        total = total + input[i][0]*input[i][0];
    }
    return std::sqrt(total/N);
}

// Spectral Bandwidth Function
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class spectral_bandwidth{
    private:
        Matrix<T,NUM_FRAMES,1> centroid;
        Matrix<T,NUM_FRAMES,1> bandwidth;
    public:
        Matrix<T,2,1> compute(Matrix<T,N,1> &input){
            input.reshape(NUM_FRAMES,NUM_BINS);
            std::size_t FFT_SIZE = (NUM_BINS-1)*2;
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T mag_total = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
                    centroid[i][0] = centroid[i][0] + input[i][j]*j;
                    mag_total = mag_total + input[i][j];
                }
                if(mag_total==0) centroid[i][0] = 0;
                else centroid[i][0] = centroid[i][0]/mag_total;
            }

            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T mag_sum = 0;
                T square = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
                    mag_sum = mag_sum + input[i][j];
                    square = square + (j-centroid[i][0])*(j-centroid[i][0])*input[i][j];
                }
                if(mag_sum==0) bandwidth = 0;
                else bandwidth = std::sqrt(square/mag_sum);
            }
            input.reset_shape();
            Matrix<T,2,1> output;
            output[0][0] = mean(bandwidth);
            output[1][0] = standard_deviation(bandwidth);
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
    private:
        Matrix<T,NUM_FRAMES,1> rolloff;
    public:
        Matrix<T,2,1> compute(Matrix<T,N,1> &input,T percentile){
            input.reshape(NUM_FRAMES,NUM_BINS);
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T mag_sum = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
                    mag_sum = mag_sum + input[i][j];
                }
                
                if(mag_sum==0) mag_sum = 1e-5;

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
            output[0][1] = standard_deviation(rolloff);
            return output;
        }
};

// Spectral Flatness Function
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class spectral_flatness{
    private:
        Matrix<T,NUM_FRAMES,1> flatness;

    public:
        Matrix<T,2,1> compute(Matrix<T,N,1> &input,T percentile){
            input.reshape(NUM_FRAMES,NUM_BINS);
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T mag_sum = 0;
                T log_sum = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
                    mag_sum = mag_sum + std::abs(input[i][j]);
                    log_sum = log_sum + std::log(std::abs(input[i][j]));
                }
                if(mag_sum==0) flatness[i][0] = 0;
                else flatness[i][0] = std::exp(log_sum)/mag_sum;
            }
            input.reset_shape();
            Matrix<T,2,1> output;
            output[0][0] = mean(flatness);
            output[0][1] = standard_deviation(flatness);
            return output;
        }
};

// Spectral Contrast Function
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class spectral_contrast{
    public:
            Matrix<T,4,1> contrast;
            Matrix<T,NUM_FRAMES,1> temp;
    public:
        Matrix<T,4,1> compute(Matrix<T,N,1> &input){
            input.reshape(NUM_FRAMES,NUM_BINS);

            std::size_t n_bands = 4;
            T band_size = NUM_BINS/n_bands;
            for(std::size_t k=0;k<n_bands;k++){
                for(std::size_t i=0;i<NUM_FRAMES;i++){
                    T temp_max = input[i][k*band_size];
                    T temp_min = input[i][k*band_size];

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
                temp.reset_shape();
                FBE[k][0] = temp/count;
            }
            input.reset_shape();
            return FBE;
        }
};

// Temporal Features Function
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class temporal_features{
    private:
        Matrix<T,NUM_FRAMES,1> frame_energy;
        Matrix<T,NUM_FRAMES-1,1> onset_strength;
    public:
        Matrix<T,5,1> compute(Matrix<T,N,1> &input){
            input.reshape(NUM_FRAMES,NUM_BINS);
            for(std::size_t i=0;i<NUM_FRAMES;i++){
                T energy = 0;
                for(std::size_t j=0;j<NUM_BINS;j++){
                    energy = energy + input[i][j]*input[i][j];
                }
                frame_energy[i][0] = energy;
            }
            for(std::size_t i=1;i<NUM_FRAMES;i++){
                onset_strength[i-1][0] = max(0,frame_energy[i]-frame_energy[i-1]);
            }
            input.reset_shape();
            Matrix<T,5,1> temporal;
            temporal[0][0] = mean(frame_energy);
            temporal[1][0] = standard_deviation(frame_energy);
            temporal[2][0] = max_value(frame_energy);
            if(NUM_FRAMES-1>0){
                temporal[3][0] = mean(onset_strength);
                temporal[4][0] = standard_deviation(onset_strength);
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
                    entropy = entropy - (input[i][j]/total_sum)*std::log(input[i][j]/total_sum);
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
            T m = mean(input);
            if(m==0) m = 1e-5;
            return max_value(input)/m;
        }
};

// MFCC Class
template <class T, std::size_t N, std::size_t NUM_FRAMES, std::size_t NUM_BINS>
class MFCC{
    public:
        // Mel Function
        // Converts Hz to Mel scale.
        T mel(T &input){
            return 2595*std::log10(1+input/700);
        }

        T invmel(T &input){
            // f(mel) = 700 * (10^(mel/2595) - 1)
            return 700*(std::pow(10,input/2595)-1);
        }

        // Mel-Frequency Cepstral Coefficients Function (MFCC)
        Matrix<T,2,1> compute(Matrix<T,N,1> &input, std::size_t sr){ // sr = sample rate
            // std::size_t n_mels = 13; // Hardcoded no of mel filter size.
            input.reshape(NUM_FRAMES,NUM_BINS);
            T f_min = 0;
            T f_max = sr/2;
            T mel_min = mel(f_min); // usually f_min = 0 Hz
            T mel_max = mel(f_max); // usually f_max = sample_rate/2
            // T mel_points = linspace(mel_min, mel_max, 13 + 2);
            Matrix<T, 13+2, 1> mel_points;
            for(std::size_t i = 0; i < 13+2; i++){
                mel_points[i][0] = mel_min + (mel_max - mel_min) * i / (13 + 1);
            }

            Matrix<T,13,NUM_BINS> H;
            std::size_t index = 0;

            Matrix<std::size_t, 13+2, 1> bin;
            for(std::size_t i = 0; i < 13+2; i++){
                T f = invmel(mel_points[i][0]);
                bin[i][0] = std::floor((NUM_BINS + 1) * f / sr);
                if(bin[i][0] >= NUM_BINS) bin[i][0] = NUM_BINS - 1;
            }

            for(std::size_t m = 0; m < 13; m++){
                std::size_t start = bin[m][0];
                std::size_t center = bin[m+1][0];
                std::size_t end = bin[m+2][0];
                for(std::size_t k = start; k <= center; k++){
                    if(center != start) 
                        H[m][k] = (T)(k - start) / (center - start);
                    else
                        H[m][k] = 1.0;
                }
                for(std::size_t k = center; k <= end; k++){
                    if(end != center)
                        H[m][k] = (T)(end - k) / (end - center);
                    else
                        H[m][k] = 1.0;
                }
            }
            Matrix<T,13,NUM_FRAMES>mel_energy;
            for(std::size_t t=0;t<NUM_FRAMES;t++){
                for(std::size_t j=0;j<13;j++){
                    mel_energy[j][t] = 0;
                    for(std::size_t k=0;k<NUM_BINS;k++){
                        mel_energy[j][t] = mel_energy[j][t] + H[j][k]*input[t][k];
                    }
                }
            }

            for(std::size_t i=0;i<13;i++){
                for(std::size_t j=0;j<NUM_FRAMES;j++){
                    mel_energy[i][j] = std::log(mel_energy[i][j] + 1e-5);
                }
            }

            Matrix<T,13*NUM_FRAMES,1> mfcc;
            mfcc.reshape(13,NUM_FRAMES);
            for(std::size_t t = 0; t < NUM_FRAMES; t++){
                for(std::size_t c = 0; c < 13; c++){
                    mfcc[c][t] = 0;
                    for(std::size_t m = 0; m < 13; m++){
                        mfcc[c][t] += mel_energy[m][t] * std::cos(M_PI * c * (m + 0.5) / 13);
                    }
                }
            }
            mfcc.reset_shape();
            input.reset_shape();
            
            Matrix<T,2,1> output;
            output[0][0] = mean(mfcc);
            output[1][0] = standard_deviation(mfcc);
            return output;
        }
};
#endif