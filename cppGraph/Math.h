//
// Created by Ben Meyers on 2/11/26.
//

#ifndef RELATIVITY_MATH_H
#define RELATIVITY_MATH_H
#include <cmath>
namespace Math {
    class Vector2 {
    public:
        float x;
        float y;

        constexpr Vector2():x(0.0f), y(0.0f){}
        constexpr Vector2(float x, float y): x(x), y(y){}

        [[nodiscard]] friend Vector2 operator+(const Vector2& a, const Vector2& b)
        {
            return {a.x + b.x, a.y + b.y};
        }

        [[nodiscard]] friend Vector2 operator-(const Vector2& a, const Vector2& b)
        {
            return {a.x - b.x, a.y - b.y};
        }

        [[nodiscard]] friend Vector2 operator*(const Vector2& a, const Vector2& b)
        {
            return {a.x * b.x, a.y * b.y};
        }

        [[nodiscard]] friend Vector2 operator*(const Vector2& a, const float& b)
        {
            return {a.x * b, a.y * b};
        }

        [[nodiscard]] friend Vector2 operator*(const float& a, const Vector2& b)
        {
            return {a * b.x, a * b.y};
        }

        Vector2& operator*= (float sc) {
            x *= sc;
            y *= sc;
            return *this;
        }

    };

    [[nodiscard]] inline float Sqrt(float value)
	{
		return sqrtf(value);
	}

    template <typename T>
    [[nodiscard]] constexpr T Max(T a, T b)
    {
        return (a < b ? b : a);
    }

    template <typename T>
    [[nodiscard]] constexpr T Min(T a, T b)
    {
        return (a < b ? a : b);
    }

    template <typename T>
    [[nodiscard]] constexpr T Clamp(T value, T lower, T upper)
    {
        return Min(upper, Max(lower, value));
    }

    [[nodiscard]] inline float Abs(float value)
    {
        return std::abs(value);
    }

    [[nodiscard]] inline bool NearlyZero(float val, float epsilon = 0.001f)
    {
        return Abs(val) <= epsilon;
    }

    [[nodiscard]] inline bool NearlyEqual(float a, float b, float epsilon = 0.001f)
    {
        return Abs(a - b) <= epsilon;
    }

    constexpr float Pi = 3.14159265f;

    [[nodiscard]] inline float Sin(float value)
    {
        return sinf(value);
    }

    [[nodiscard]] inline float Cos(float value)
    {
        return cosf(value);
    }

    [[nodiscard]] inline float Acos(float value)
    {
        return acosf(value);
    }

    [[nodiscard]] inline float Pow(float base, float exp)
    {
        return powf(base, exp);
    }

    [[nodiscard]] inline float ToDegrees(float radians)
{
    return radians * 180.0f / 3.14159265f;
}

}
#endif //RELATIVITY_MATH_H