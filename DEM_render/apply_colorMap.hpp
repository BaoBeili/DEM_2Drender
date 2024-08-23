//
//  apply_colorMap.hpp
//  DEM_render
//
//  Created by huanghongli on 2024/7/17.
//

#ifndef apply_colorMap_hpp
#define apply_colorMap_hpp
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <map>
#include <stdio.h>
#include <opencv2/opencv.hpp>

// RGBColor给定的构造函数，在初始化的时候将输入的red，green，blue的值
struct RGBColor {
    int r, g, b;
    RGBColor(int red = 0, int green = 0, int blue = 0) : r(red), g(green), b(blue) {}
};

struct TColorItem {
    float m_fValue;
    int m_rgb[3];

    TColorItem() {}

    TColorItem(float fV, int r, int g, int b)
        : m_fValue(fV) {
        m_rgb[0] = r;
        m_rgb[1] = g;
        m_rgb[2] = b;
    }

    TColorItem(float fV, const RGBColor& rgb)
        : m_fValue(fV) {
        m_rgb[0] = rgb.r;
        m_rgb[1] = rgb.g;
        m_rgb[2] = rgb.b;
    }
    
    // 带 RGBColor 参数的构造函数，用于初始化 m_fValue 和 m_rgb 数组
    RGBColor toColor() const {
        return RGBColor(m_rgb[0], m_rgb[1], m_rgb[2]);
    }
    
    // 输出流，输出结果的值
    friend std::ostream& operator<<(std::ostream& os, const TColorItem& item) {
        os << "Value: " << item.m_fValue << ", RGB: ("
           << item.m_rgb[0] << ", " << item.m_rgb[1] << ", " << item.m_rgb[2] << ")";
        return os;
    }
};

typedef std::vector<TColorItem> VColorItem;

struct TColorMap{
    std::string    mColorModel;//颜色模式, "RGB", "HSV"
    VColorItem mItems; // 储存颜色色卡
    bool       mHasForeg;
    bool       mHasBackg;
    bool       mHasNull;
    float      mClrForeground[3];//前景颜色
    float      mClrBackground[3];//背景颜色
    float      mClrNull[3];      //空值颜色
    float      mMin;             //最小值
    float      mMax;             //最大值
    static std::map<std::string, RGBColor> s_clrNames;
    
    friend std::ostream& operator<<(std::ostream& os, const TColorMap& map) {
        os << "Color Model: " << map.mColorModel << "\n"
           << "Min Value: " << map.mMin << ", Max Value: " << map.mMax << "\n"
           << "Has Foreground: " << std::boolalpha << map.mHasForeg << "\n"
           << "Foreground Color: (" << map.mClrForeground[0] << ", " << map.mClrForeground[1] << ", " << map.mClrForeground[2] << ")\n"
           << "Has Background: " << std::boolalpha << map.mHasBackg << "\n"
           << "Background Color: (" << map.mClrBackground[0] << ", " << map.mClrBackground[1] << ", " << map.mClrBackground[2] << ")\n"
           << "Has Null: " << std::boolalpha << map.mHasNull << "\n"
           << "Null Color: (" << map.mClrNull[0] << ", " << map.mClrNull[1] << ", " << map.mClrNull[2] << ")\n"
           << "Items:\n";
        for (const auto& item : map.mItems) {
            os << item << "\n";
        }
        return os;
    }
    
    TColorMap();
};

// 定义用于比较 TColorItem 对象的谓词函数
struct ColorItemComparer
{
    bool operator()(const TColorItem& item1, const TColorItem& item2) const
    {
        return (item1.m_rgb[0] == item2.m_rgb[0]) &&
               (item1.m_rgb[1] == item2.m_rgb[1]) &&
               (item1.m_rgb[2] == item2.m_rgb[2]);
    }
};

// 定义一个ColorStop的结构体用于储存色卡
struct ColorStop{
    float percentage;
    uint8_t r;
    uint8_t g;
    uint8_t b;
    public:
        ColorStop(float percentage, uint8_t r, uint8_t g,
                  uint8_t b);

        std::vector<float> getColor()const;
    
    // 定义输出流运算符
    friend std::ostream& operator<<(std::ostream& os, const ColorStop& cs) {
        os << "ColorStop(percentage: " << cs.percentage
           << ", r: " << static_cast<int>(cs.r)
           << ", g: " << static_cast<int>(cs.g)
           << ", b: " << static_cast<int>(cs.b) << ")";
        return os;
    }
    
};

// 读取色卡函数，返回TColorMap
bool parserCptFile(const std::string& sFilePath, TColorMap& clrMap);

/**
 * @brief linearGradient 线性渐变插值函数
 * @param stops 颜色转折点列表，按位置百分比排序
 * @param interpPercentage 插值位置百分比（小数）
 * @return 插值结果，为4个元素的std::vector<float>
 */
std::vector<float> linearGradient(const std::vector<ColorStop>& stops, float interpPercentage);
/**
 * @brief RGB2HSV函数
 * @brief HSV2RGB函数
 */
cv::Vec3f rgb2hsv(const cv::Vec3f& rgb);
cv::Vec3f hsv2rgb(const cv::Vec3f& hsv);
cv::Vec3f gmt_illuminate(const cv::Vec3f& rgb, float intensity,
                         float color_hsv_max_s, float color_hsv_max_v,
                         float color_hsv_min_s, float color_hsv_min_v);
#endif /* apply_colorMap_hpp */
