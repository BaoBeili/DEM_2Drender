#ifndef GRID_N_HPP
#define GRID_N_HPP

#include <string>

// 定义结构体 N
struct N_model {
    int mode; // 模式
    double sigma; // sigma参数
    double ambient; // -N参数偏移
};

// 定义一个结构体
struct Ctrl {
    double azimuth;          // 第一个参数 投影角度
    std::string CPTfiles;    // 设置颜色色卡
    N_model N;
    double a;
    // 默认构造函数
    Ctrl();
};

struct GridHeader {
    int n_columns;
    int n_rows;
    int mx;
    double* wesn;
    double z_max;
    double z_min;
};

struct Grid {
    GridHeader* header;
    float* data;
};

// 计算行列号对应的索引
int gmt_M_ijp(GridHeader* header, int row, int col);

// 声明cosd函数
double cosd(double degree);

// 声明gmt_M_is_fnan函数
bool gmt_M_is_fnan(float value);
#endif // GRID_N_HPP
