//
// Created by Ben Meyers on 2/18/26.
//
#pragma once
#include <utility>


template<size_t R, size_t C>
class Matrix {
    float mData[R][C];

public:
    template<size_t A, size_t B, size_t D>
    static constexpr bool Compatible(const Matrix<A,B>& a, const Matrix<B,D>& b) {
        return true;
    }
    Matrix(): mData{}{}
    [[nodiscard]] constexpr std::pair<size_t, size_t> Shape() const {
        return {R, C};
    }
    [[nodiscard]] constexpr float Data(size_t r, size_t c)const{return mData[r][c];}
    constexpr float& Data(size_t r, size_t c){return mData[r][c];}
    [[nodiscard]] constexpr size_t Rows() const { return R; }
    [[nodiscard]] constexpr size_t Cols() const { return C; }

    template<size_t M>
    Matrix<R,M> operator*(Matrix<C,M> other){
        Matrix<R,M> result{};
        // for each row in this
        for (size_t i = 0; i < R; i++){
            // for each column in other
            for (size_t j = 0; j < M; j++){
                // for this.col/other.row
                for (size_t k = 0; k < C; k++){
                    // this           = this[row][col] * other[otherRow][otherCol]
                    result.Data(i,j) += mData[i][k] * other.Data(k,j);
                }
            }
        }
        return result;
    }

    Matrix<R,C> operator*(float scalar){
        Matrix<R,C> result{};
        for (size_t i = 0; i < R; i++){
            for (size_t j = 0; j < C; j++){
                result.Data(i, j) = mData[i][j] * scalar;
            }
        }
        return result;
    }

    Matrix<R,C> operator/(float scalar){
        return *this * (1.0f/scalar);
    }

    Matrix<R,C> operator+(Matrix<R,C> other){
        Matrix<R,C> result{};
        for (size_t i = 0; i < R; i++){
            for (size_t j = 0; j < C; j++){
                result.Data(i, j) = mData[i][j] + other.Data(i, j);
            }
        }
        return result;
    }

    Matrix<R,C> operator-(Matrix<R,C> other){
        Matrix<R,C> result{};
        for (size_t i = 0; i < R; i++){
            for (size_t j = 0; j < C; j++){
                result.Data(i, j) = mData[i][j] - other.Data(i, j);
            }
        }
        return result;
    }

    Matrix<R,C> operator+(float bias){
        Matrix<R,C> result{};
        for (size_t i = 0; i < R; i++){
            for (size_t j = 0; j < C; j++){
                result.Data(i, j) = mData[i][j] + bias;
            }
        }
        return result;
    }

    Matrix<R,C> operator-(float bias){
        return *this + (-1.0f * bias);
    }
};



