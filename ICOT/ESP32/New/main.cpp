#include <iostream>
#include <bits/stdc++.h>
typedef size_t ll;
using namespace std;

#include "Matrix.hpp"
#include "feature.hpp"
#include "audio_cleaner.hpp"
#include "stft.hpp"
#include "dwt.hpp"

int main(){
	ll t = 0;
    cin>>t;
    while(t--){
        // Features
        // MFCC<float,100,5,20>sc_object;
        // // spectral_crest_factor<float,100>sc_object;
        // const ll n = 32000;
        // Matrix<float,n,1> s;
        // for(ll i=0;i<n;i++) cin>>s[i][0];
        // Matrix<float,5,1> buffer1;
        // Matrix<float,5,1> buffer2;
        // Matrix<float,20,1> buffer3;
        // Matrix<float,4,1> onset_strength; // NUM_FRAMES - 1
        // // auto final = sc_object.compute(s);
        // s.reset_shape();
        // apply_bandpass_filter(s);
        // // clip(s,(float)-100,(float)100);
        // // auto final = s;
        // // cout<<final;
        // // Matrix<float,2,1> final;
        // // log_thingy(s);
        // // final = sc_object.compute(s,10);
        // // final[0][0] = skewness(s,mean(s),standard_deviation(s,mean(s)));
        // // final[1][0] = kurtosis(s,mean(s),standard_deviation(s,mean(s)));
        // // cout<<final;
        // for(ll i=0;i<s.rows();i++) cout<<s[i][0]<<" ";
        // // for(ll i=0;i<final.rows();i++) printf("%.1f ",final[i][0]);
        // cout<<endl;


        // DWT
        // const ll n = 32038;
        // Matrix<float,n,1> s;
        // for(ll i=0;i<32000;i++) cin>>s[i][0];
        // for(ll i=32000;i<n;i++) s[i][0] = 0;
        // dwt<float,32000,6>dwt_object;
        // dwt_object.compute(s);
        // zoom_dwt<float,n,7,16003,128,100> zoom_object;
        // // log_thingy(s); // Dont do this in esp32, it is already implemented in zoom function.
        // auto final = zoom_object.zoom(s);
        // final.reset_shape();

        // for(ll i=0;i<final.rows();i++) cout<<final[i][0]<<" ";
        // cout<<endl;

        // STFT
        const ll n=(1+32000/256)*513;
        const ll NUM_FRAMES = (1+32000/256);
        const ll NUM_BINS = 513;
        Matrix<float,n,1> s;
        for(ll i=0;i<32000;i++) cin>>s[i][0];
        for(ll i=32000;i<n;i++) s[i][0] = 0;
        stft<float,32000,1024,256,n> stft_object;
        Matrix<float,NUM_FRAMES*NUM_BINS,1> output;
        stft_object.compute(s,output);
        output.reset_shape();
        log_thingy(output);
        zoom_stft<float,NUM_FRAMES*NUM_BINS,NUM_FRAMES,NUM_BINS,100,128> zoom;
        auto final = zoom.zoom(output);
        normalize_mean_std(final,mean(final),standard_deviation(final,mean(final)));
        for(ll i=0;i<100*128;i++) cout<<final[i][0]<<" ";
        cout<<endl;
    }
}