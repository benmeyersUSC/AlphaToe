//
// Created by Ben Meyers on 2/18/26.
//

#include "DynamicMatrix.h"
#include <stdexcept>
#include <string>
#include <functional>
DynamicMatrix::DynamicMatrix(const size_t rows, const size_t cols, const float init)
    : mData(rows * cols, init), mRows(rows), mCols(cols) {}

float& DynamicMatrix::at(size_t r, size_t c) {
    // slick
    return mData[r * mCols + c];
}

float DynamicMatrix::at(const size_t r, const size_t c) const {
    return mData[r * mCols + c];
}

// matmul
DynamicMatrix DynamicMatrix::operator*(const DynamicMatrix& other) const {
    // assure dimensions are compatible
    if (mCols != other.mRows)
    {
        throw std::runtime_error("Matrix multiply: incompatible shapes (" +
            std::to_string(mRows) + "x" + std::to_string(mCols) + ") * (" +
            std::to_string(other.mRows) + "x" + std::to_string(other.mCols) + ")");
    }

    // build result matrix
    DynamicMatrix result(mRows, other.mCols);
    // for each row
    for (size_t i = 0; i < mRows; i++)
    {
        // iterate over other's columns
        for (size_t j = 0; j < other.mCols; j++)
        {
            // and finally the shared dimension (their rows, our cols)
            for (size_t k = 0; k < mCols; k++)
            {
                // dot product
                // k is iterating RIGHT on our row, DOWN on their col
                result.at(i, j) += at(i, k) * other.at(k, j);
            }
        }
    }
    return result;
}

// matrix addition
DynamicMatrix DynamicMatrix::operator+(const DynamicMatrix& other) const {
    if (mRows != other.mRows || mCols != other.mCols) {
        throw std::runtime_error("Matrix add: incompatible shapes (" +
            std::to_string(mRows) + "x" + std::to_string(mCols) + ") + (" +
            std::to_string(other.mRows) + "x" + std::to_string(other.mCols) + ")");
    }

    // simple add
    DynamicMatrix result(mRows, mCols);
    for (size_t i = 0; i < mRows; i++)
        for (size_t j = 0; j < mCols; j++) {
            result.at(i, j) = at(i, j) + other.at(i, j);
        }
    return result;
}

// scalar mult
DynamicMatrix DynamicMatrix::operator*(float scalar) const {
    DynamicMatrix result(mRows, mCols);
    for (size_t i = 0; i < mRows; i++)
        for (size_t j = 0; j < mCols; j++) {
            result.at(i, j) = at(i, j) * scalar;
        }
    return result;
}

DynamicMatrix DynamicMatrix::Transpose() const {
    DynamicMatrix result(mCols, mRows);
    for (size_t r = 0; r < mRows; r++)
        for (size_t c = 0; c < mCols; c++)
            {
            result.at(c, r) = at(r, c);
        }
    return result;
}

DynamicMatrix DynamicMatrix::ElemWiseMult(const DynamicMatrix& other) const {
    if (mRows != other.mRows || mCols != other.mCols)
        {throw std::runtime_error("HadamardProduct: incompatible shapes");}
    DynamicMatrix result(mRows, mCols);
    for (size_t i = 0; i < mRows; i++) {
        for (size_t j = 0; j < mCols; j++) {
            result.at(i, j) = at(i, j) * other.at(i, j);
        }
    }
    return result;
}

DynamicMatrix DynamicMatrix::operator-(const DynamicMatrix& other) const {
    if (mRows != other.mRows || mCols != other.mCols) {
        throw std::runtime_error("Matrix subtract: incompatible shapes");
    }
    DynamicMatrix result(mRows, mCols);
    for (size_t i = 0; i < mRows; i++) {
        for (size_t j = 0; j < mCols; j++) {
            result.at(i, j) = at(i, j) - other.at(i, j);
        }
    }
    return result;
}

// apply a function to each value
DynamicMatrix DynamicMatrix::Apply(const std::function<float(float)>& fn) const {
    DynamicMatrix result(mRows, mCols);
    std::transform(mData.begin(), mData.end(), result.mData.begin(), fn);
    return result;
}
