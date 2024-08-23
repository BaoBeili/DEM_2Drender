//
//  grid_N.cpp
//  DEM_render
//
//  Created by huanghongli on 2024/7/16.
//

#include "grid_N.hpp"


// 构造函数实现
Ctrl::Ctrl() : azimuth(270.0), CPTfiles(""), N{0,1.0,0.0},a(1.0) {
    // 可以在这里添加其他初始化逻辑
}

bool gmt_M_is_fnan(float value) {
    return std::isnan(value);
};

int gmt_M_ijp(GridHeader* header, int row, int col) {
    return row * header->n_columns + col;
};

double cosd(double degree) {
    return cos(degree * M_PI / 180.0);
}


