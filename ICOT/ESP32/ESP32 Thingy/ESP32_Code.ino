#include <WiFi.h>
#include <WiFiClient.h>
#include <driver/i2s.h>
#include <Arduino.h>
// SET_LOOP_TASK_STACK_SIZE(131072)
SET_LOOP_TASK_STACK_SIZE(32768)
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

// #include "esp_task_wdt.h"
#include "esp_system.h"
#define YIELD() taskYIELD();
#include <algorithm>

/* ================= WIFI CONFIG ================= */
const char* ssid = "Sachcith";
const char* password = "7639404514";

const char* SERVER_HOST = "192.168.27.187";
const uint16_t SERVER_PORT = 5000;

const char* SERVER_CMD = "/command";
const char* SERVER_UPLOAD = "/upload";

/* ================= AUDIO CONFIG ================= */
const uint32_t SAMPLE_RATE = 16000;
#define RECORD_SECONDS 2
#define TOTAL_SAMPLES (SAMPLE_RATE * RECORD_SECONDS)

#define POLL_MS 1500
#define BUFFER_SAMPLES 256

/* ================= I2S PINS ================= */
#define I2S_WS 45
#define I2S_SD 35
#define I2S_SCK 37
#define I2S_PORT I2S_NUM_0

/* ================= GLOBAL BUFFERS ================= */
int32_t i2s_buffer[BUFFER_SAMPLES];
int16_t pcm_buffer[BUFFER_SAMPLES];
// int16_t audio[TOTAL_SAMPLES];

static Matrix<float, 32038, 1> signal_matrix;

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

// void record_audio() {

//   Serial.println("Recording 2 seconds...");

//   uint32_t samples_recorded = 0;

//   while (samples_recorded < TOTAL_SAMPLES) {

//     size_t bytes_read = 0;
//     i2s_read(I2S_PORT, i2s_buffer, sizeof(i2s_buffer), &bytes_read, 20 / portTICK_PERIOD_MS);

//     int samples = bytes_read / 4;
//     YIELD();

//     for (int i = 0; i < samples && samples_recorded < TOTAL_SAMPLES; i++) {
//       int32_t s = i2s_buffer[i] >> 8;
//       float val = (float)(s >> 8);
//       // audio[samples_recorded++] = (float)val / 32768.0f;
//       signal_matrix[samples_recorded++][0] = val;
//     }

//     vTaskDelay(1);
//   }

//   Serial.println("Recording done.");
// }

bool record_audio() {

  int seconds = 2;  // Force 30 seconds total

  uint32_t total_samples = SAMPLE_RATE * seconds;
  uint32_t data_bytes = total_samples * 2;

  uint32_t samples_sent = 0;
  uint32_t solenoid_off_sample = SAMPLE_RATE * 15;  // 15 sec mark

  while (samples_sent < total_samples) {
    Serial.println(samples_sent);
    size_t bytes_read = 0;
    esp_err_t res = i2s_read(I2S_PORT, i2s_buffer, sizeof(i2s_buffer), &bytes_read, portMAX_DELAY);
    // esp_err_t res = i2s_read(I2S_PORT, i2s_buffer, sizeof(i2s_buffer), &bytes_read, 20 / portTICK_PERIOD_MS);
    if (res != ESP_OK || bytes_read == 0) {
        Serial.println("I2S read timeout");
        vTaskDelay(10);
        continue;
    }

    int samples = bytes_read / 4;

    for (int i = 0; i < samples && samples_sent < total_samples; i++) {
      int32_t s = i2s_buffer[i] >> 8;
      pcm_buffer[i] = (int16_t)(s >> 8);
      signal_matrix[samples_sent + i][0] = (float)pcm_buffer[i];
      delay(1);
    }
    delay(1);
    samples_sent += samples;

    // After 15 seconds turn OFF solenoid
    if (samples_sent >= solenoid_off_sample) {
    }
  }

  // =========================
  // FINISH
  // =========================

  // while (client.connected()) {
  //   String line = client.readStringUntil('\n');
  //   if (line == "\r") break;
  // }

  Serial.println("30s Audio uploaded (15s ON + 15s OFF)");
  // client.stop();
  return true;
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
  client.println("X-API-KEY: esp32_secret_key_123");   // ADD THIS
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
  client.println("X-API-KEY: esp32_secret_key_123");   // ADD THIS
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
// static Matrix<float,32038,1> signal; // Change this so it will work for both dwt and stft.
void run_pipeline() {
    Serial.println("1");
    constexpr size_t FINAL_SIZE = 32038; //compute_final_size(32000,6,8);
    // constexpr size_t TOTAL_SAMPLES = 32000;
    constexpr size_t sr = 16000;
    constexpr size_t FFT_SIZE = 1024;
    constexpr size_t HOP_SIZE = 256;
    constexpr size_t NUM_FRAMES = 1 + (TOTAL_SAMPLES) / HOP_SIZE;
    constexpr size_t NUM_BINS = FFT_SIZE/2 + 1;
    constexpr size_t FINAL = FINAL_SIZE;// max(max((NUM_FRAMES+1)*NUM_BINS,TOTAL_SAMPLES),FINAL_SIZE);
    // esp_task_wdt_deinit();   // 🔴 disable watchdog
    static Matrix<float,FINAL,1> signal; // Change this so it will work for both dwt and stft.
    // for(size_t i=0;i<FINAL;i++){
    //     signal[i][0] = 0;
    // }

    delay(10);
    // for(size_t i=0;i<TOTAL_SAMPLES;i++){
    //     signal[i][0] = signal[i][0];
    //     if(i%1023==0) YIELD();
    // }
    Serial.println("11");

    for (size_t i = 0; i < TOTAL_SAMPLES; i += 512) {
        size_t end = i + 512;
        if(end>=TOTAL_SAMPLES) end = TOTAL_SAMPLES;
        for (size_t j = i; j < end; j++) {
            signal[j][0] = signal_matrix[j][0];
        }
        delay(10);   // give scheduler a break
    }
    Serial.println("12");

    Serial.println("12");
    apply_pre_emphasis(signal,static_cast<float>(0.97));
    //apply_bandpass_filter(signal);
    normalize_rms(signal);

    static Matrix<float,NUM_FRAMES*NUM_BINS,1>stft_output_actual;
    stft<float,TOTAL_SAMPLES,FFT_SIZE,HOP_SIZE,FINAL_SIZE> stft_object;
    stft_object.compute(signal,stft_output_actual);
    
    stft_output_actual.reset_shape();
    log_thingy(stft_output_actual);
    stft_output_actual.reshape(NUM_FRAMES,NUM_BINS);
    
    zoom_stft<float,NUM_FRAMES*NUM_BINS,NUM_FRAMES,NUM_BINS,100,128> zoom_stft_object;
    static Matrix<float,100*128,1> stft_output = zoom_stft_object.zoom(stft_output_actual);
    
    stft_output.reset_shape();
    float mean_stft = mean(stft_output);
    float std_stft = standard_deviation(stft_output,mean_stft);

    normalize_mean_std(stft_output,mean_stft,std_stft);
    
    nan_inf_values(stft_output);
    clip(stft_output,static_cast<float>(-100),static_cast<float>(100));

    constexpr size_t N = 100*128;
    constexpr size_t r = 100;
    constexpr size_t c = 128;

    static Matrix<float,100*128,1> stft_positive;

    for(size_t i=0;i<N;i++){
        stft_positive[i][0] = abs_error(stft_output[i][0]);
    }

    static Matrix<float,r,1> buffer1;
    static Matrix<float,r,1> buffer2;
    static Matrix<float,35,1> features_stft;
    static Matrix<float,2,1> temp;

    Serial.println("2");
    stft_output.reset_shape();
    features_stft[0][0] = mean(stft_output);
    features_stft[1][0] = standard_deviation(stft_output,features_stft[0][0]);
    features_stft[2][0] = max_value(stft_output);
    features_stft[3][0] = min_value(stft_output);
    features_stft[4][0] = percentile(stft_output,features_stft[3][0],features_stft[2][0],static_cast<size_t>(0.25*NUM_FRAMES*NUM_BINS));
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
    static Matrix<float,4,1> constrast = spectral_contrast_object.compute(stft_positive,buffer1);
    features_stft[14][0] = constrast[0][0];
    features_stft[15][0] = constrast[1][0];
    features_stft[16][0] = constrast[2][0];
    features_stft[17][0] = constrast[3][0];

    features_stft[18][0] = zcr(stft_output,r,c);

    frequency_band_energies<float,N,r,c>frequency_band_energies_object;
    static Matrix<float,5,1> fb_energies = frequency_band_energies_object.compute(stft_positive);
    features_stft[19][0] = fb_energies[0][0];
    features_stft[20][0] = fb_energies[1][0];
    features_stft[21][0] = fb_energies[2][0];
    features_stft[22][0] = fb_energies[3][0];
    features_stft[23][0] = fb_energies[4][0];

    temporal_features<float,N,r,c>temporal_features_object;
    static Matrix<float,r-1,1> onset_strength;
    static Matrix<float,5,1> temporal = temporal_features_object.compute(stft_output,buffer1,onset_strength);
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

    // disp(features_stft);
    Serial.println("3");

    dwt<float,TOTAL_SAMPLES,6> dwt_object;
    signal.reset_shape();
    for(size_t i=0;i<signal.rows();i++){
        signal[i][0] = abs_error(signal[i][0]);
    }
    log_thingy(signal);
    signal.reset_shape();

    zoom_dwt<float,FINAL,7,16003,128,100> zoom_dwt_object;
    static Matrix<float,100*128,1> dwt_output = zoom_dwt_object.zoom(signal);
    dwt_output.reset_shape();
    // for(size_t i=0;i<dwt_output.rows();i++){
    //     dwt_output[i][0] = abs_error(dwt_output[i][0]);
    // }
    // log_thingy(dwt_output);
    
    // dwt_output.reset_shape();
    // float mean_dwt = mean(dwt_output);
    // float std_dwt = standard_deviation(dwt_output,mean_dwt);

    // normalize_mean_std(dwt_output,mean_dwt,std_dwt);
    
    nan_inf_values(dwt_output);
    clip(dwt_output,static_cast<float>(-100),static_cast<float>(100));

    // constexpr size_t N = 100*128;

    static Matrix<float,100*128,1> dwt_positive;

    for(size_t i=0;i<N;i++){
        dwt_positive[i][0] = abs_error(dwt_output[i][0]);
    }

    for(size_t i=0;i<r;i++) buffer1[i][0] = 0;
    for(size_t i=0;i<r;i++) buffer2[i][0] = 0;

    static Matrix<float,35,1> features_dwt;
    // static Matrix<float,2,1> temp;

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
    Serial.println("4");

    // spectral_rolloff<float,N,r,c>spectral_rolloff_object;
    temp = spectral_rolloff_object.compute(dwt_positive,static_cast<float>(0.85),buffer1);
    features_dwt[10][0] = temp[0][0];
    features_dwt[11][0] = temp[1][0];

    // spectral_flatness<float,N,r,c>spectral_flatness_object;
    temp = spectral_flatness_object.compute(dwt_positive,buffer1);
    features_dwt[12][0] = temp[0][0];
    features_dwt[13][0] = temp[1][0];

    // spectral_contrast<float,N,r,c>spectral_contrast_object;
    // static Matrix<float,4,1> constrast = spectral_contrast_object.compute(dwt_positive);
    constrast = spectral_contrast_object.compute(dwt_positive,buffer1);
    features_dwt[14][0] = constrast[0][0];
    features_dwt[15][0] = constrast[1][0];
    features_dwt[16][0] = constrast[2][0];
    features_dwt[17][0] = constrast[3][0];

    features_dwt[18][0] = zcr(dwt_output,r,c);

    // frequency_band_energies<float,N,r,c>frequency_band_energies_object;
    // static Matrix<float,5,1> fb_energies = frequency_band_energies_object.compute(dwt_positive);
    fb_energies = frequency_band_energies_object.compute(dwt_positive);
    features_dwt[19][0] = fb_energies[0][0];
    features_dwt[20][0] = fb_energies[1][0];
    features_dwt[21][0] = fb_energies[2][0];
    features_dwt[22][0] = fb_energies[3][0];
    features_dwt[23][0] = fb_energies[4][0];

    // temporal_features<float,N,r,c>temporal_features_object;
    // static Matrix<float,5,1> temporal = temporal_features_object.compute(dwt_output);
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

    // disp(features_dwt);

    static Matrix<float,70,1> final_features;
    // static Matrix<float,35,1> features_stft;
    for(size_t i=0;i<35;i++){
        final_features[i][0] = features_stft[i][0];
        final_features[i+35][0] = features_dwt[i][0];
    }


    Serial.println("5");
    Dense<float,64,70> hidden_1;
    Dense<float,32,64> hidden_2;
    Dense<float,6,32> hidden_3;

    static Matrix<float,64,1> h1_output;
    hidden_1.forward(final_features,mlp_weights::hidden_1_weights,mlp_weights::hidden_1_bias,h1_output);
    relu(h1_output);
    static Matrix<float,32,1> h2_output;
    hidden_2.forward(h1_output,mlp_weights::hidden_2_weights,mlp_weights::hidden_2_bias,h2_output);
    relu(h2_output);
    static Matrix<float,6,1> h3_output;
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

  // send_wav();
  // delay(500);
  // send_result(final_answer);
}

/* ================================================== */
/* ================= SETUP ========================== */
/* ================================================== */

void wifi_connect() {

  Serial.println("Connecting WiFi...");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);   // IMPORTANT: prevents WiFi sleep disconnects
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  WiFi.setSleep(false);

  Serial.println("\nWiFi connected");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
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


// Initialize Task WDT with 60s timeout
// void init_task_wdt() {
//     esp_task_wdt_config_t config = {};
//     config.timeout_ms = 120 * 1000;    // 60 seconds
//     config.idle_core_mask = (1 << 0) | (1 << 1); // monitor both CPU cores
//     config.trigger_panic = true;      // causes reset on timeout

//     // esp_err_t err = esp_task_wdt_init(&config);
//     if (err != ESP_OK) {
//         printf("TWDT init failed: %d\n", err);
//     }
// }



void i2s_task(void* arg) {

  Serial.println("I2S task running on Core:");
  Serial.println(xPortGetCoreID());

  while (true) {

    // Your audio capture
    record_audio();

    // Your DSP + ML
    run_pipeline();

    delay(200);
    // IMPORTANT → yield to system
    vTaskDelay(1);
  }
}

void i2s_stop_safe() {
  i2s_stop(I2S_PORT);
  i2s_driver_uninstall(I2S_PORT);
  delay(50);  // let hardware settle
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Serial.println("Setup running on Core:");
  // Serial.println(xPortGetCoreID());

  // wifi_connect();
  // delay(200);
  // WiFi.setSleep(false);

  // ✅ INIT I2S FIRST
  // if (!i2s_init()) {
  //   Serial.println("I2S init failed");
  //   while (1);
  // }

  // // ✅ THEN start task
  // xTaskCreatePinnedToCore(
  //   i2s_task,
  //   "i2s_task",
  //   32768,   // increase stack
  //   NULL,
  //   1,
  //   NULL,
  //   1
  // );

  Serial.println("Setup done");
}

/* ================================================== */
/* ================= LOOP =========================== */
/* ================================================== */

// static WiFiClient client;
// void loop() {

//   Serial.println("=== NEW CYCLE ===");

//   // ======================
//   // 1. INIT MIC
//   // ======================
//   Serial.println("Starting I2S...");
//   if (!i2s_init()) {
//     Serial.println("I2S init failed");
//     delay(2000);
//     return;
//   }

//   delay(100);  // stabilize

//   // ======================
//   // 2. RECORD
//   // ======================
//   Serial.println("Recording...");
//   record_audio();

//   // ======================
//   // 3. STOP MIC
//   // ======================
//   Serial.println("Stopping I2S...");
//   i2s_stop_safe();

//   delay(100);  // IMPORTANT

//   // ======================
//   // 4. RUN PIPELINE
//   // ======================
//   Serial.println("Running pipeline...");
  
//   // OPTIONAL but STRONGLY recommended:
//   // esp_task_wdt_deinit();   // 🔴 prevent reset during heavy compute
  
//   run_pipeline();
  
//   // esp_task_wdt_init(10, true);  // ✅ re-enable

//   // ======================
//   // 5. WAIT
//   // ======================
//   Serial.println("Cooling down...");
//   delay(5000);
// }
// void loop() {
//   Serial.println("Hello Loop");
//   delay(2000);
//   run_pipeline();
// }
// #include "esp_timer.h"

// void loop() {
//   Serial.println("Hello Loop");
//   delay(2000);

//   int64_t start = esp_timer_get_time();  // microseconds

//   run_pipeline();

//   int64_t end = esp_timer_get_time();

//   Serial.print("Pipeline time: ");
//   Serial.print((end - start) / 1000.0);
//   Serial.println(" ms");
// }

#include "esp_timer.h"
#include "esp_heap_caps.h"

#define RUNS 100

double times[RUNS];

void compute_stats() {
  // sort (simple bubble since RUNS small)
  for (int i = 0; i < RUNS - 1; i++) {
    for (int j = 0; j < RUNS - i - 1; j++) {
      if (times[j] > times[j + 1]) {
        double t = times[j];
        times[j] = times[j + 1];
        times[j + 1] = t;
      }
    }
  }

  double min_t = times[0];
  double max_t = times[RUNS - 1];

  double sum = 0;
  for (int i = 0; i < RUNS; i++) sum += times[i];
  double mean = sum / RUNS;

  double median = (RUNS % 2 == 0) ?
    (times[RUNS/2 - 1] + times[RUNS/2]) / 2.0 :
    times[RUNS/2];

  // std deviation
  double var = 0;
  for (int i = 0; i < RUNS; i++) {
    var += (times[i] - mean) * (times[i] - mean);
  }
  var /= RUNS;
  double stddev = sqrt(var);

  double p95 = times[(int)(0.95 * RUNS)];

  Serial.println("\n===== BENCHMARK RESULTS =====");
  Serial.printf("Min: %.3f ms\n", min_t);
  Serial.printf("Max: %.3f ms\n", max_t);
  Serial.printf("Mean: %.3f ms\n", mean);
  Serial.printf("Median: %.3f ms\n", median);
  Serial.printf("Std Dev: %.3f ms\n", stddev);
  Serial.printf("95th %%: %.3f ms\n", p95);
}

void loop() {

  Serial.println("Starting benchmark in 3 sec...");
  delay(3000);

  size_t heap_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
  size_t min_heap_before = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);

  for (int i = 0; i < RUNS; i++) {

    int64_t start = esp_timer_get_time();

    run_pipeline();

    int64_t end = esp_timer_get_time();

    times[i] = (end - start) / 1000.0;

    Serial.printf("Run %d: %.3f ms\n", i, times[i]);

    delay(50);  // small cooldown
  }

  size_t heap_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
  size_t min_heap_after = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);

  compute_stats();

  Serial.println("\n===== MEMORY STATS =====");
  Serial.printf("Heap before: %u bytes\n", heap_before);
  Serial.printf("Heap after: %u bytes\n", heap_after);
  Serial.printf("Min heap before: %u bytes\n", min_heap_before);
  Serial.printf("Min heap after: %u bytes\n", min_heap_after);

  // Stack usage (important!)
  Serial.printf("Stack high water mark: %u bytes\n",
    uxTaskGetStackHighWaterMark(NULL));

  while (true); // stop after one benchmark
}
