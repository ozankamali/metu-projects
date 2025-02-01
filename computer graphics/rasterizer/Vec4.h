#ifndef __VEC4_H__
#define __VEC4_H__
#define NO_COLOR -1

class Vec4
{
public:
    double x, y, z, w;
    int colorId;

    Vec4();
    Vec4(double x, double y, double z, double w);
    Vec4(double x, double y, double z, double w, int colorId);
    Vec4(const Vec4 &other);

    double getNthComponent(int n);

    friend std::ostream &operator<<(std::ostream &os, const Vec4 &v);
};

#endif