import numpy as np

def STFT(Normalized_data, sample_rate=48000, frame_size=2048, hop_length=512, n_fft=2048, window_type="hamming", spectrum_type="power", scaling="log", keep_positive_freq=True, center=False):

    Ndata = Normalized_data
    No_of_frames = 1 + int(np.floor((len(Normalized_data) - frame_size)/hop_length))
    
    # print(No_of_frames)

    Sliced_NData_frames = []

    temp = 0
    for _ in range(No_of_frames):
        k = Ndata[temp:temp+frame_size]
        Sliced_NData_frames.append(k)
        temp += hop_length

    # print(len(k))

    N = frame_size
    for i in range(len(Sliced_NData_frames)):

        # HWindow = 0.54 - 0.46*np.cos(2*np.pi*[j for j in range(N)]/N)
        HWindow = 0.54 - 0.46*np.cos(2*np.pi*np.arange(N)/N)

        Sliced_NData_frames[i] *= HWindow

    # STFT_Data = []
    # k1 = 0
    # for i1 in range(No_of_frames):
    #     temp1 = Sliced_NData_frames[i1]
    #     temp2 = []
    #     for i2 in range(n_fft):
    #         k1 += temp1[i2]*np.exp((-np.j*2*np.pi*i1*i2)/n_fft)
    #         temp2.append(k1)
    #     STFT_Data.append(temp2)

    STFT_Data = []
    for frame in Sliced_NData_frames:

        if N < n_fft:
            padded_frame = np.zeros(n_fft)
            padded_frame[:N] = frame
        else:
            padded_frame = frame

        X = []
        for k in range(n_fft):

            cp_n = 0 + 0j

            for n in range(len(padded_frame)):

                cp_n += padded_frame[n]*np.exp(-1j*2*np.pi*k*n/n_fft)

            X.append(cp_n)
        X = np.array(X)

        if keep_positive_freq:
            X = X[:n_fft//2 + 1]

        if spectrum_type == "power":
            P = np.abs(X)**2
        else:
            P = np.abs(X)
        
        if scaling == "log":
            P = np.log(P + 1e-10)
        
        STFT_Data.append(P)
    

    return np.array(STFT_Data)