#ifndef Matrix_H
#define Matrix_H
#include <cstddef>
#include <iostream>

template<class T, std::size_t R, std::size_t C>
class Matrix {
private:
    T data[R * C];

    bool reshaped = false;
    std::size_t runtime_rows = R;
    std::size_t runtime_cols = C;

public:

    // Activate reshape view
    void reshape(std::size_t newR, std::size_t newC) {
        // You ensure newR * newC == R*C
        /*
        if(newR*newC != R*C){
            std::cout<<"Invalid Size"<<std::endl;
            return;
        }
        */
        reshaped = true;
        runtime_rows = newR;
        runtime_cols = newC;
    }

    // Reset to original shape
    void reset_shape() {
        reshaped = false;
        runtime_rows = R;
        runtime_cols = C;
    }

    T* operator[](std::size_t i) {
        std::size_t cols = reshaped ? runtime_cols : C;
        return data + i * cols;
    }

    const T* operator[](std::size_t i) const {
        std::size_t cols = reshaped ? runtime_cols : C;
        return data + i * cols;
    }

    std::size_t rows() const {
        return reshaped ? runtime_rows : R;
    }

    std::size_t cols() const {
        return reshaped ? runtime_cols : C;
    }
};

#endif


/*#ifndef Matrix_H
#define Matrix_H
#include <cstddef>

template<class T, std::size_t R, std::size_t C>
class Matrix {
    private:
        T data[R * C];

    public:
        T* operator[](std::size_t i) {
            return data + i * C;
        }

        const T* operator[](std::size_t i) const {
            return data + i * C;
        }

        constexpr std::size_t rows() const { return R; }
        constexpr std::size_t cols() const { return C; }
};

#endif*/