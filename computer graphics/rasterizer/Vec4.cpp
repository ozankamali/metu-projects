#include <iomanip>
#include "Vec4.h"

Vec4::Vec4()
{
    this->x = 0.0;
    this->y = 0.0;
    this->z = 0.0;
    this->w = 0.0;
    this->colorId = NO_COLOR;
}

Vec4::Vec4(double x, double y, double z, double w)
{
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
    this->colorId = NO_COLOR;
}

Vec4::Vec4(double x, double y, double z, double w, int colorId)
{
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
    this->colorId = colorId;
}

Vec4::Vec4(const Vec4 &other)
{
    this->x = other.x;
    this->y = other.y;
    this->z = other.z;
    this->w = other.w;
    this->colorId = other.colorId;
}

double Vec4::getNthComponent(int n)
{
    switch (n)
    {
    case 0:
        return this->x;

    case 1:
        return this->y;

    case 2:
        return this->z;

    case 3:
    default:
        return this->w;
    }
}

std::ostream &operator<<(std::ostream &os, const Vec4 &v)
{
    os << std::fixed << std::setprecision(6) << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
    return os;
}