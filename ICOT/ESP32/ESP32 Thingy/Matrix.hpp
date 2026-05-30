#ifndef Matrix_H
#define Matrix_H
#include <cstddef>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "esp_heap_caps.h" // ESP32 PSRAM allocation
#include "esp_task_wdt.h"
#define YIELD() delay(1)

template<class T, std::size_t R, std::size_t C>
class Matrix {
private:
    T* data = nullptr; // pointer to hold the matrix data in PSRAM
    bool reshaped = false;
    std::size_t runtime_rows = R;
    std::size_t runtime_cols = C;

public:
    // Constructor: allocate PSRAM
    Matrix() {
        // data = (T*)heap_caps_malloc(sizeof(T) * R * C, MALLOC_CAP_SPIRAM);
        data = (T*)heap_caps_malloc(sizeof(T) * R * C, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!data) {
            Serial.println("PSRAM allocation failed!");
            std::abort();
        }
        delay(1);
        // std::memset(data, 0, sizeof(T) * R * C);
        // for (size_t i = 0; i < R * C; i++) {
        //     data[i] = 0;
        //     if ((i & 255) == 0) {
        //         esp_task_wdt_reset();
        //         taskYIELD();
        //     }
        // }
    }

    // Copy constructor (deep copy)
    Matrix(const Matrix& other) {
        runtime_rows = other.runtime_rows;
        runtime_cols = other.runtime_cols;
        reshaped = other.reshaped;
        data = (T*)heap_caps_malloc(sizeof(T) * R * C, MALLOC_CAP_SPIRAM);
        if (!data) {
            std::cerr << "PSRAM allocation failed!" << std::endl;
            std::abort();
        }
        for (size_t i = 0; i < R * C; i++) {
            data[i] = other.data[i];
            if ((i & 1023) == 0) delay(0);
        }
    }

    // Copy assignment (deep copy)
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            if (data) heap_caps_free(data);
            runtime_rows = other.runtime_rows;
            runtime_cols = other.runtime_cols;
            reshaped = other.reshaped;
            data = (T*)heap_caps_malloc(sizeof(T) * R * C, MALLOC_CAP_SPIRAM);
            if (!data) {
                std::cerr << "PSRAM allocation failed!" << std::endl;
                std::abort();
            }
            for (size_t i = 0; i < R * C; i++) {
                data[i] = other.data[i];
                if ((i & 1023) == 0) delay(0);
            }
        }
        return *this;
    }

    // Move constructor
    Matrix(Matrix&& other) noexcept {
        data = other.data;
        reshaped = other.reshaped;
        runtime_rows = other.runtime_rows;
        runtime_cols = other.runtime_cols;
        other.data = nullptr;
    }

    // Move assignment
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            if (data) heap_caps_free(data);
            data = other.data;
            reshaped = other.reshaped;
            runtime_rows = other.runtime_rows;
            runtime_cols = other.runtime_cols;
            other.data = nullptr;
        }
        return *this;
    }

    // Destructor: free PSRAM
    ~Matrix() {
        if (data) {
            heap_caps_free(data);
            data = nullptr;
        }
    }

    // Reshape view (does not change data)
    void reshape(std::size_t newR, std::size_t newC) {
        reshaped = true;
        runtime_rows = newR;
        runtime_cols = newC;
    }

    void reset_shape() {
        reshaped = false;
        runtime_rows = R;
        runtime_cols = C;
    }

    // Access operators
    T* operator[](std::size_t i) {
        std::size_t cols = reshaped ? runtime_cols : C;
        return data + i * cols;
    }

    const T* operator[](std::size_t i) const {
        std::size_t cols = reshaped ? runtime_cols : C;
        return data + i * cols;
    }

    std::size_t rows() const { return reshaped ? runtime_rows : R; }
    std::size_t cols() const { return reshaped ? runtime_cols : C; }

    // Optional: fill function
    void fill(T value) {
        constexpr size_t CHUNK = 512;

        for (std::size_t i = 0; i < R * C; i++) {
            data[i] = value;

            if ((i % CHUNK) == 0) {
                // esp_task_wdt_reset();
                taskYIELD();
            }
        }
    }
};

#endif