#ifndef Matrix_H
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

#endif