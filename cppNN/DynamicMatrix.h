//
// Created by Ben Meyers on 2/18/26.
//
#pragma once


#include <vector>
#include <functional>

class DynamicMatrix {
    std::vector<float> mData;
    size_t mRows, mCols;

public:
    DynamicMatrix(size_t rows, size_t cols, float init = 0.0f);

    float& at(size_t r, size_t c);
    [[nodiscard]] float at(size_t r, size_t c) const;

    [[nodiscard]] size_t Rows() const { return mRows; }
    [[nodiscard]] size_t Cols() const { return mCols; }

    DynamicMatrix operator*(const DynamicMatrix& other) const;
    DynamicMatrix operator+(const DynamicMatrix& other) const;
    DynamicMatrix operator*(float scalar) const;

    DynamicMatrix Apply(const std::function<float(float)>& fn) const;

    [[nodiscard]] DynamicMatrix Transpose() const;
    [[nodiscard]] DynamicMatrix ElemWiseMult(const DynamicMatrix& other) const;
    DynamicMatrix operator-(const DynamicMatrix& other) const;
};

