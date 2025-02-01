#ifndef __MATRIX4_H__
#define __MATRIX4_H__

class Matrix4
{
public:
    double values[4][4];

    Matrix4();
    Matrix4(double values[4][4]);
    Matrix4(const Matrix4 &other);

    double get(int row, int col);
    void set(int row, int col, double value);
    friend std::ostream &operator<<(std::ostream &os, const Matrix4 &m);
};

#endif