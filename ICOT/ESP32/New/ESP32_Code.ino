#include <WiFi.h>
#include <WiFiClient.h>
#include <driver/i2s.h>
#include <Arduino.h>
#include <esp_dsp.h>

#include "Matrix.hpp"
// #include "disp.hpp"
#include "Dense.hpp"
#include "activation.hpp"
#include "stft.hpp"
#include "dwt.hpp"
#include "feature.hpp"
#include "audio_cleaner.hpp"
#include "mlp_model_weights.cc"

#include <algorithm>

/* ================= WIFI CONFIG ================= */
const char* ssid = "Sachcith";
const char* password = "7639404514";

const char* SERVER_HOST = "192.168.3.187";
const uint16_t SERVER_PORT = 5000;

/* ================= AUDIO CONFIG ================= */
const uint32_t SAMPLE_RATE = 16000;
#define RECORD_SECONDS 1
#define TOTAL_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)

#define BUFFER_SAMPLES 256

/* ================= I2S PINS ================= */
#define I2S_WS 25
#define I2S_SD 33
#define I2S_SCK 32
#define I2S_PORT I2S_NUM_0

/* ================= GLOBAL BUFFERS ================= */
int32_t i2s_buffer[BUFFER_SAMPLES];
// int16_t audio[TOTAL_SAMPLES];

static Matrix<float, 16038, 1> signal_matrix;

/* ================================================== */
/* ================= WAV HEADER ===================== */
/* ================================================== */

void write_wav_header(uint8_t* header, uint32_t total_data_bytes) {
  uint32_t file_size_minus8 = total_data_bytes + 36;
  uint32_t byte_rate = SAMPLE_RATE * 1 * 2;
  uint16_t block_align = 2;
  uint16_t bits = 16;
  uint16_t channels = 1;

  memcpy(header + 0, "RIFF", 4);
  memcpy(header + 4, &file_size_minus8, 4);
  memcpy(header + 8, "WAVE", 4);
  memcpy(header + 12, "fmt ", 4);

  uint32_t subchunk1 = 16;
  uint16_t audio_format = 1;

  memcpy(header + 16, &subchunk1, 4);
  memcpy(header + 20, &audio_format, 2);
  memcpy(header + 22, &channels, 2);
  memcpy(header + 24, &SAMPLE_RATE, 4);
  memcpy(header + 28, &byte_rate, 4);
  memcpy(header + 32, &block_align, 2);
  memcpy(header + 34, &bits, 2);

  memcpy(header + 36, "data", 4);
  memcpy(header + 40, &total_data_bytes, 4);
}

/* ================================================== */
/* ================= I2S INIT ======================= */
/* ================================================== */

bool i2s_init() {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_SAMPLES,
    .use_apll = false
  };

  i2s_pin_config_t pins = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  if (i2s_driver_install(I2S_PORT, &cfg, 0, NULL) != ESP_OK) return false;
  if (i2s_set_pin(I2S_PORT, &pins) != ESP_OK) return false;

  i2s_start(I2S_PORT);
  return true;
}

/* ================================================== */
/* ================= RECORD AUDIO =================== */
/* ================================================== */

void record_audio() {

  Serial.println("Recording 2 seconds...");

  uint32_t samples_recorded = 0;

  while (samples_recorded < TOTAL_SAMPLES) {

    size_t bytes_read = 0;
    i2s_read(I2S_PORT, i2s_buffer, sizeof(i2s_buffer), &bytes_read, portMAX_DELAY);

    int samples = bytes_read / 4;

    for (int i = 0; i < samples && samples_recorded < TOTAL_SAMPLES; i++) {
      int32_t s = i2s_buffer[i] >> 8;
      float val = (float)(s >> 8);
      // audio[samples_recorded++] = (float)val / 32768.0f;
      signal_matrix[samples_recorded++][0] = val;
    }
  }

  Serial.println("Recording done.");
}

/* ================================================== */
/* ================= SEND WAV ======================= */
/* ================================================== */

void send_wav() {

  WiFiClient client;
  if (!client.connect(SERVER_HOST, SERVER_PORT)) {
    Serial.println("WAV upload failed");
    return;
  }

  uint32_t data_bytes = TOTAL_SAMPLES * 2;

  client.printf("POST /upload HTTP/1.1\r\n");
  client.printf("Host: %s\r\n", SERVER_HOST);
  client.println("Content-Type: audio/wav");
  client.printf("Content-Length: %u\r\n", data_bytes + 44);
  client.println("Connection: close");
  client.println();

  uint8_t header[44];
  write_wav_header(header, data_bytes);
  client.write(header, 44);

  for (uint32_t i = 0; i < TOTAL_SAMPLES; i++) {
    int16_t s = (int16_t)(signal_matrix[i][0] * 32767.0f);
    client.write((uint8_t*)&s, 2);
  }

  client.stop();
  Serial.println("WAV sent.");
}

/* ================================================== */
/* ================= SEND RESULT ==================== */
/* ================================================== */

void send_result(int class_id) {

  WiFiClient client;
  if (!client.connect(SERVER_HOST, SERVER_PORT)) {
    Serial.println("Result send failed");
    return;
  }

  String body = "{\"class\": " + String(class_id) + "}";

  client.printf("POST /result HTTP/1.1\r\n");
  client.printf("Host: %s\r\n", SERVER_HOST);
  client.println("Content-Type: application/json");
  client.printf("Content-Length: %d\r\n", body.length());
  client.println("Connection: close");
  client.println();
  client.print(body);

  client.stop();
  Serial.println("Result sent.");
}

/* ================================================== */
/* ================= DSP + MLP ====================== */
/* ================================================== */

void run_pipeline() {

  constexpr size_t FFT_SIZE = 1024;
  constexpr size_t HOP_SIZE = 512;
  constexpr size_t FINAL_SIZE = 16038;


  // constexpr size_t TOTAL_SAMPLES = 32000;
  constexpr size_t sr = 16000;
  constexpr size_t NUM_FRAMES = (TOTAL_SAMPLES - FFT_SIZE) / HOP_SIZE;
  constexpr size_t NUM_BINS = FFT_SIZE/2 + 1;
  constexpr size_t FINAL =
    std::max(
        std::max(static_cast<size_t>((NUM_FRAMES + 1) * NUM_BINS),
                 static_cast<size_t>(TOTAL_SAMPLES)),
        static_cast<size_t>(FINAL_SIZE)
    );


  // for (size_t i = 0; i < TOTAL_SAMPLES; i++)
  //   signal_matrix[i][0] = audio[i];

  apply_pre_emphasis(signal_matrix, 0.97f);
  normalize_rms(signal_matrix);



  stft<float, TOTAL_SAMPLES, FFT_SIZE, HOP_SIZE, FINAL_SIZE> stft_obj;
  stft_obj.compute(signal_matrix);
  log_thingy(signal_matrix);
  
  zoom_stft<float,16038,NUM_FRAMES,NUM_BINS,100,128> zoom_stft_object;
    static Matrix<float,100*128,1> stft_output;
    zoom_stft_object.zoom(signal_matrix,stft_output);
    
    stft_output.reset_shape();
    float mean_stft = mean(stft_output);
    float std_stft = standard_deviation(stft_output,mean_stft);

    normalize_mean_std(stft_output,mean_stft,std_stft);
    
    nan_inf_values(stft_output);
    clip(stft_output,static_cast<float>(-100),static_cast<float>(100));

    constexpr size_t N = 100*128;

    static Matrix<float,35,1> features_stft;
    static Matrix<float,2,1> temp;

    temporal_features<float,N,NUM_FRAMES,NUM_BINS>temporal_features_object;
    static Matrix<float,5,1> temporal = temporal_features_object.compute(stft_output);
    features_stft[24][0] = temporal[0][0];
    features_stft[25][0] = temporal[1][0];
    features_stft[26][0] = temporal[2][0];
    features_stft[27][0] = temporal[3][0];
    features_stft[28][0] = temporal[4][0];

    features_stft[29][0] = skewness(stft_output,features_stft[0][0],features_stft[1][0]);
    features_stft[30][0] = kurtosis(stft_output,features_stft[0][0],features_stft[1][0]);


    stft_output.reset_shape();
    features_stft[0][0] = mean(stft_output);
    features_stft[1][0] = standard_deviation(stft_output,features_stft[0][0]);
    features_stft[2][0] = max_value(stft_output);
    features_stft[3][0] = min_value(stft_output);
    features_stft[4][0] = percentile(stft_output,features_stft[3][0],features_stft[2][0],static_cast<size_t>(0.25*TOTAL_SAMPLES));
    features_stft[5][0] = percentile(stft_output,features_stft[3][0],features_stft[2][0],static_cast<size_t>(0.75*TOTAL_SAMPLES));

    features_stft[18][0] = zcr(stft_output);

    // static Matrix<float,100*128,1> stft_output;

    for(size_t i=0;i<N;i++){
        stft_output[i][0] = abs_error(stft_output[i][0]);
    }

    spectral_centroid<float,N,NUM_FRAMES,NUM_BINS>spectral_centroid_object;
    temp = spectral_centroid_object.compute(stft_output,sr);
    features_stft[6][0] = temp[0][0];
    features_stft[7][0] = temp[1][0];

    spectral_bandwidth<float,N,NUM_FRAMES,NUM_BINS>spectral_bandwidth_object;
    temp = spectral_bandwidth_object.compute(stft_output);
    features_stft[8][0] = temp[0][0];
    features_stft[9][0] = temp[1][0];

    spectral_rolloff<float,N,NUM_FRAMES,NUM_BINS>spectral_rolloff_object;
    temp = spectral_rolloff_object.compute(stft_output,static_cast<float>(0.85));
    features_stft[10][0] = temp[0][0];
    features_stft[11][0] = temp[1][0];

    spectral_flatness<float,N,NUM_FRAMES,NUM_BINS>spectral_flatness_object;
    temp = spectral_flatness_object.compute(stft_output);
    features_stft[12][0] = temp[0][0];
    features_stft[13][0] = temp[1][0];

    spectral_contrast<float,N,NUM_FRAMES,NUM_BINS>spectral_contrast_object;
    static Matrix<float,4,1> constrast = spectral_contrast_object.compute(stft_output);
    features_stft[14][0] = constrast[0][0];
    features_stft[15][0] = constrast[1][0];
    features_stft[16][0] = constrast[2][0];
    features_stft[17][0] = constrast[3][0];

    frequency_band_energies<float,N,NUM_FRAMES,NUM_BINS>frequency_band_energies_object;
    static Matrix<float,5,1> fb_energies = frequency_band_energies_object.compute(stft_output);
    features_stft[19][0] = fb_energies[0][0];
    features_stft[20][0] = fb_energies[1][0];
    features_stft[21][0] = fb_energies[2][0];
    features_stft[22][0] = fb_energies[3][0];
    features_stft[23][0] = fb_energies[4][0];

    MFCC<float,N,NUM_FRAMES,NUM_BINS>MFCC_object;
    temp = MFCC_object.compute(stft_output,sr);
    features_stft[31][0] = temp[0][0];
    features_stft[32][0] = temp[1][0];

    spectral_entropy_feature<float,N,NUM_FRAMES,NUM_BINS>spectral_entropy_feature_object;
    features_stft[33][0] = spectral_entropy_feature_object.compute(stft_output);

    spectral_crest_factor<float,N>spectral_crest_factor_object;
    features_stft[34][0] = spectral_crest_factor_object.compute(stft_output);
    
    // disp(features_stft);

    dwt<float,TOTAL_SAMPLES,6> dwt_object;
    dwt_object.compute(signal_matrix);
    log_thingy(signal_matrix);

    zoom_dwt<float,16038,7,16003,100,128> zoom_dwt_object;
    // static Matrix<float,100*128,1> stft_output = zoom_dwt_object.zoom(signal_matrix);
    zoom_dwt_object.zoom(signal_matrix,stft_output);
    
    stft_output.reset_shape();
    float mean_dwt = mean(stft_output);
    float std_dwt = standard_deviation(stft_output,mean_dwt);

    normalize_mean_std(stft_output,mean_dwt,std_dwt);
    
    nan_inf_values(stft_output);
    clip(stft_output,static_cast<float>(-100),static_cast<float>(100));

    // constexpr size_t N = 100*128;

    // static Matrix<float,100*128,1> stft_output;


    static Matrix<float,35,1> features_dwt;
    // static Matrix<float,2,1> temp;

    stft_output.reset_shape();
    features_dwt[0][0] = mean(stft_output);
    features_dwt[1][0] = standard_deviation(stft_output,features_dwt[0][0]);
    features_dwt[2][0] = max_value(stft_output);
    features_dwt[3][0] = min_value(stft_output);
    features_dwt[4][0] = percentile(stft_output,features_dwt[3][0],features_dwt[2][0],static_cast<size_t>(0.25*TOTAL_SAMPLES));
    features_dwt[5][0] = percentile(stft_output,features_dwt[3][0],features_dwt[2][0],static_cast<size_t>(0.75*TOTAL_SAMPLES));

    features_dwt[18][0] = zcr(stft_output);

    // temporal_features<float,N,NUM_FRAMES,NUM_BINS>temporal_features_object;
    // static Matrix<float,5,1> temporal = temporal_features_object.compute(stft_output);
    temporal = temporal_features_object.compute(stft_output);
    features_dwt[24][0] = temporal[0][0];
    features_dwt[25][0] = temporal[1][0];
    features_dwt[26][0] = temporal[2][0];
    features_dwt[27][0] = temporal[3][0];
    features_dwt[28][0] = temporal[4][0];

    features_dwt[29][0] = skewness(stft_output,features_dwt[0][0],features_dwt[1][0]);
    features_dwt[30][0] = kurtosis(stft_output,features_dwt[0][0],features_dwt[1][0]);


    for(size_t i=0;i<N;i++){
        stft_output[i][0] = abs_error(stft_output[i][0]);
    }
    // spectral_centroid<float,N,NUM_FRAMES,NUM_BINS>spectral_centroid_object;
    temp = spectral_centroid_object.compute(stft_output,sr);
    features_dwt[6][0] = temp[0][0];
    features_dwt[7][0] = temp[1][0];

    // spectral_bandwidth<float,N,NUM_FRAMES,NUM_BINS>spectral_bandwidth_object;
    temp = spectral_bandwidth_object.compute(stft_output);
    features_dwt[8][0] = temp[0][0];
    features_dwt[9][0] = temp[1][0];

    // spectral_rolloff<float,N,NUM_FRAMES,NUM_BINS>spectral_rolloff_object;
    temp = spectral_rolloff_object.compute(stft_output,static_cast<float>(0.85));
    features_dwt[10][0] = temp[0][0];
    features_dwt[11][0] = temp[1][0];

    // spectral_flatness<float,N,NUM_FRAMES,NUM_BINS>spectral_flatness_object;
    temp = spectral_flatness_object.compute(stft_output);
    features_dwt[12][0] = temp[0][0];
    features_dwt[13][0] = temp[1][0];

    // spectral_contrast<float,N,NUM_FRAMES,NUM_BINS>spectral_contrast_object;
    // static Matrix<float,4,1> constrast = spectral_contrast_object.compute(stft_output);
    constrast = spectral_contrast_object.compute(stft_output);
    features_dwt[14][0] = constrast[0][0];
    features_dwt[15][0] = constrast[1][0];
    features_dwt[16][0] = constrast[2][0];
    features_dwt[17][0] = constrast[3][0];


    // frequency_band_energies<float,N,NUM_FRAMES,NUM_BINS>frequency_band_energies_object;
    // static Matrix<float,5,1> fb_energies = frequency_band_energies_object.compute(stft_output);
    fb_energies = frequency_band_energies_object.compute(stft_output);
    features_dwt[19][0] = fb_energies[0][0];
    features_dwt[20][0] = fb_energies[1][0];
    features_dwt[21][0] = fb_energies[2][0];
    features_dwt[22][0] = fb_energies[3][0];
    features_dwt[23][0] = fb_energies[4][0];

    // MFCC<float,N,NUM_FRAMES,NUM_BINS>MFCC_object;
    temp = MFCC_object.compute(stft_output,sr);
    features_dwt[31][0] = temp[0][0];
    features_dwt[32][0] = temp[1][0];

    // spectral_entropy_feature<float,N,NUM_FRAMES,NUM_BINS>spectral_entropy_feature_object;
    features_dwt[33][0] = spectral_entropy_feature_object.compute(stft_output);

    // spectral_crest_factor<float,N>spectral_crest_factor_object;
    features_dwt[34][0] = spectral_crest_factor_object.compute(stft_output);
    
    // disp(features_dwt);

  static Matrix<float,70,1> final_features;

  // >>> YOU FILL THIS BLOCK <<<

    // Matrix<float,35,1> features_stft;
    for(size_t i=0;i<35;i++){
        final_features[i][0] = features_stft[i][0];
        final_features[i+35][0] = features_dwt[i][0];
    }


    Dense<float,64,70> hidden_1;
    Dense<float,32,64> hidden_2;
    Dense<float,6,32> hidden_3;

    Matrix<float,64,1> h1_output;
    hidden_1.forward(final_features,mlp_weights::hidden_1_weights,mlp_weights::hidden_1_bias,h1_output);
    relu(h1_output);
    Matrix<float,32,1> h2_output;
    hidden_2.forward(h1_output,mlp_weights::hidden_2_weights,mlp_weights::hidden_2_bias,h2_output);
    relu(h2_output);
    Matrix<float,6,1> h3_output;
    hidden_3.forward(h2_output,mlp_weights::student_output_weights,mlp_weights::student_output_bias,h3_output);
    softmax(h3_output);

    float temp_final_for_answer = -1;
    size_t final_answer;
    for(size_t i=0;i<h3_output.rows();i++){
        if(temp_final_for_answer<h3_output[i][0]){
            temp_final_for_answer = h3_output[i][0];
            final_answer = i+1;
        }
    }

  Serial.print("Class: ");
  Serial.println(final_answer);

  send_wav();
  delay(500);
  send_result(final_answer);
}

/* ================================================== */
/* ================= SETUP ========================== */
/* ================================================== */

void wifi_connect() {

  Serial.println("Connecting WiFi...");

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
}

void wifi_off() {

  Serial.println("Turning WiFi OFF");

  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);

  delay(200);
} 


// void setup() {

//   Serial.begin(115200);
//   delay(1000);

//   WiFi.begin(ssid, password);
//   while (WiFi.status() != WL_CONNECTED) delay(300);

//   Serial.println("WiFi connected");

//   dsps_fft2r_init_fc32(NULL, 1024);

//   if (!i2s_init()) {
//     Serial.println("I2S init failed");
//     while (1);
//   }

//   Serial.println("System Ready");
// }

void setup() {

  Serial.begin(115200);
  delay(1000);

  wifi_connect();

  dsps_fft2r_init_fc32(NULL, 1024);

  if (!i2s_init()) {
    Serial.println("I2S init failed");
    while (1);
  }

  Serial.println("System Ready");
}

/* ================================================== */
/* ================= LOOP =========================== */
/* ================================================== */

void loop() {

  WiFiClient client;

  if (!client.connect(SERVER_HOST, SERVER_PORT)) {
    delay(1500);
    return;
  }

  client.printf("GET /command HTTP/1.1\r\n");
  client.printf("Host: %s\r\n", SERVER_HOST);
  client.println("Connection: close");
  client.println();

  while (client.connected()) {
    String line = client.readStringUntil('\n');
    if (line == "\r") break;
  }

  String payload = client.readString();
  client.stop();

  if (payload.indexOf("\"record\":true") != -1 ||
      payload.indexOf("\"record\": true") != -1) {

    Serial.println("Server requested recording");

    /* RECORD AUDIO */
    record_audio();

    /* TURN OFF WIFI DURING HEAVY DSP */
    wifi_off();

    /* RUN DSP + MLP */
    run_pipeline();

    /* RECONNECT WIFI */
    wifi_connect();
  }

  delay(1500);
}

// void loop() {

//   WiFiClient client;

//   if (!client.connect(SERVER_HOST, SERVER_PORT)) {
//     delay(1500);
//     return;
//   }

//   client.printf("GET /command HTTP/1.1\r\n");
//   client.printf("Host: %s\r\n", SERVER_HOST);
//   client.println("Connection: close");
//   client.println();

//   while (client.connected()) {
//     String line = client.readStringUntil('\n');
//     if (line == "\r") break;
//   }

//   String payload = client.readString();
//   client.stop();

//   if (payload.indexOf("\"record\":true") != -1 ||
//       payload.indexOf("\"record\": true") != -1) {

//     Serial.println("Server requested recording");

//     record_audio();
//     run_pipeline();
//   }

//   delay(1500);
// }
