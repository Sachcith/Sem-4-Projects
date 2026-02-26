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
template <class T, std::size_t N>
Matrix<T,2,1> spectral_bandwidth(Matrix<T,N,1> &input, std::size_t NUM_FRAMES, std::size_t NUM_BINS){
    input.reshape(NUM_FRAMES,NUM_BINS);
    Matrix<T,NUM_FRAMES,1> centroid;
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

    Matrix<T,NUM_FRAMES,1> bandwidth;
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
    output[1][0] = std(bandwidth);
    return output;
}

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
template <class T, std::size_t N>
Matrix<T,2,1> spectral_rolloff(Matrix<T,N,1> &input,T percentile, std::size_t NUM_FRAMES, std::size_t NUM_BINS){
    input.reshape(NUM_FRAMES,NUM_BINS);
    Matrix<T,NUM_FRAMES,1> rolloff;
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
    output[0][1] = std(rolloff);
    return output;
}

// Spectral Flatness Function
template <class T, std::size_t N>
Matrix<T,2,1> spectral_flatness(Matrix<T,N,1> &input,T percentile, std::size_t NUM_FRAMES, std::size_t NUM_BINS){
    input.reshape(NUM_FRAMES,NUM_BINS);
    Matrix<T,NUM_FRAMES,1> flatness;
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
    output[0][1] = std(flatness);
    return output;
}

// Spectral Contrast Function
template <class T, std::size_t N>
Matrix<T,4,1> spectral_contrast(Matrix<T,N,1> &input, std::size_t NUM_FRAMES, std::size_t NUM_BINS){
    input.reshape(NUM_FRAMES,NUM_BINS);

    std::size_t n_bands = 4;
    T band_size = NUM_BINS/n_bands;
    Matrix<T,4,1> contrast;
    for(std::size_t k=0;k<n_bands;k++){
        Matrix<T,NUM_FRAMES,1> temp;
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

// Frequency Band Energy Function
template <class T, std::size_t N>
Matrix<T,5,1> frequency_band_energies(Matrix<T,N,1> &input, std::size_t NUM_FRAMES, std::size_t NUM_BINS){
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
#endif