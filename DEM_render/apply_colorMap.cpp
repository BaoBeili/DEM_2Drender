//
//  apply_colorMap.cpp
//  DEM_render
//
//  Created by huanghongli on 2024/7/17.
//

#include "apply_colorMap.hpp"


ColorStop::ColorStop(float percentage, uint8_t r, uint8_t g,
                     uint8_t b):percentage(percentage),r(r),g(g),b(b){
    
};

std::vector<float>ColorStop::getColor() const{
    return std::vector<float> {r/255.0f, g/255.0f, b/255.0f};
}


std::map<std::string, RGBColor> TColorMap::s_clrNames;
TColorMap::TColorMap():mHasForeg(false)
    , mHasBackg(false)
    , mHasNull(false)
    , mColorModel("RGB")
    , mMin(99999999999)
    , mMax(-99999999999){
};

bool parserCptFile(const std::string& sFilePath, TColorMap& clrMap){
    bool bRet = false;
    std::ifstream file(sFilePath);
    if (!file.is_open()) {
        return bRet;
    }
    
    clrMap.mItems.clear();
    
    std::string line;
    bool bStart = false;
    while (std::getline(file, line)) {
        if (!bStart) {
            if (line.at(0) == '#') {
                if (line.find("COLOR_MODEL") != std::string::npos) {
                    std::istringstream ss(line);
                    std::string key, value;
                    std::getline(ss, key, '=');
                    std::getline(ss, value);
                    if (!value.empty()) {
                        clrMap.mColorModel = value;
                        bStart = true;
                    }
                }
                continue;
            } else {
                bStart = true;
            }
        }
        // 正式内容
        if (!line.empty() && line[0] != '#') {
            std::replace(line.begin(), line.end(), '/', ' ');
            std::istringstream ss(line);
            std::vector<std::string> tokens;
            std::string token;
            while (ss >> token) {
                tokens.push_back(token);
            }

            if (tokens.size() >= 8) {
                if (clrMap.mColorModel.find("HSV") != std::string::npos) {
                    // HSV to RGB conversion is skipped for simplicity
                    // Add your HSV to RGB conversion logic here if needed
                } else {
                    TColorItem clrA(std::stof(tokens[0]), std::stoi(tokens[1]), std::stoi(tokens[2]), std::stoi(tokens[3]));
                    TColorItem clrB(std::stof(tokens[4]), std::stoi(tokens[5]), std::stoi(tokens[6]), std::stoi(tokens[7]));
                    clrMap.mItems.push_back(clrA);
                    clrMap.mItems.push_back(clrB);
                }
            } else if (tokens.size() == 4) {
                if (tokens[0] == "F") {
                    clrMap.mHasForeg = true;
                    for (int i = 0; i < 3; ++i) {
                        clrMap.mClrForeground[i] = std::stoi(tokens[i + 1]);
                    }
                } else if (tokens[0] == "B") {
                    clrMap.mHasBackg = true;
                    for (int i = 0; i < 3; ++i) {
                        clrMap.mClrBackground[i] = std::stoi(tokens[i + 1]);
                    }
                } else if (tokens[0] == "N") {
                    clrMap.mHasNull = true;
                    for (int i = 0; i < 3; ++i) {
                        clrMap.mClrNull[i] = std::stoi(tokens[i + 1]);
                    }
                } else {
                    if (TColorMap::s_clrNames.size()) {
                        TColorItem clrA(std::stof(tokens[0]), TColorMap::s_clrNames[tokens[1]]);
                        TColorItem clrB(std::stof(tokens[2]), TColorMap::s_clrNames[tokens[3]]);
                        clrMap.mItems.push_back(clrA);
                        clrMap.mItems.push_back(clrB);
                    }
                }
            }
        }
    }
    if (!clrMap.mItems.empty()) {
        if (clrMap.mItems.begin()->m_fValue > clrMap.mItems.rbegin()->m_fValue) {
            clrMap.mMin = clrMap.mItems.rbegin()->m_fValue;
            clrMap.mMax = clrMap.mItems.begin()->m_fValue;
        } else {
            clrMap.mMin = clrMap.mItems.begin()->m_fValue;
            clrMap.mMax = clrMap.mItems.rbegin()->m_fValue;
        }
    }
    
    return !clrMap.mItems.empty();
}

// 线性渲染函数
std::vector<float> linearGradient(const std::vector<ColorStop>& stops, float interpPercentage) {
    if (stops.empty()) {
        throw std::runtime_error("Unexpected stops for linear gradient.");
    }
    if (stops.size() == 1) {
        return stops[0].getColor();
    }
    std::vector<float> color{0.0f, 0.0f, 0.0f};
    bool foundInside = false;
    for (size_t i = 0; i < stops.size(); ++i) {
        const ColorStop& cur = stops[i];
        if (interpPercentage > cur.percentage) continue;
        if (i == 0) {
            color = cur.getColor();
        } else {
            const auto& last = stops[i - 1];
            const auto lastColor = last.getColor();
            const auto curColor = cur.getColor();
            for (size_t j = 0; j < color.size(); ++j) {
                color[j] = (curColor[j] - lastColor[j]) *
                           (interpPercentage - last.percentage) /
                           (cur.percentage - last.percentage) +
                           lastColor[j];
            }
        }
        foundInside = true;
        break;
    }
    if (!foundInside) {
        return stops.back().getColor();
    }
    return color;
}

/**
 * @brief: rgb2hsv
 * @param: v为RGB归一化后最大值
 * @param: S为RGB归一化后最大值
 */

cv::Vec3f rgb2hsv(const cv::Vec3f& rgb) {
    float r = rgb[0], g = rgb[1], b = rgb[2];
    float cmax = std::max(r, std::max(g, b));
    float cmin = std::min(r, std::min(g, b));
    float delta = cmax - cmin;

    float h = 0, s = 0, v = 0;
    // Compute hue
    if (delta != 0) {
        if (cmax == r)
            h = 60 * fmod((g - b) / delta, 6);
        else if (cmax == g)
            h = 60 * ((b - r) / delta + 2);
        else
            h = 60 * ((r - g) / delta + 4);
    }
    if (cmax != 0)
        s = delta / cmax;
    v = cmax;
    if (h<0){
        h = h + 360;
    }

    return cv::Vec3f(h / 360.0f, s, v);
}

cv::Vec3f hsv2rgb(const cv::Vec3f& hsv) {
    float h = hsv[0] * 360.0f, s = hsv[1], v = hsv[2];
    float hi = floor(h/60.0f);
    float f = h/60.0f - hi;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f)*s);

    float r = 0, g = 0, b = 0;
    if (hi == 0) {
        r = v;
        g = t;
        b = p;
    } else if (hi == 1) {
        r = q;
        g = v;
        b = p;
    } else if (hi == 2) {
        r = p;
        g = v;
        b = t;
    } else if (hi == 3) {
        r = p;
        g = q;
        b = v;
    } else if (hi == 4) {
        r = t;
        g = p;
        b = v;
    } else if (hi == 5) {
        r = v;
        g = p;
        b = q;
    }
    
    return cv::Vec3f(r, g, b);
}

cv::Vec3f gmt_illuminate(const cv::Vec3f& rgb, float intensity,
                         float color_hsv_max_s = 1.0, float color_hsv_max_v = 1.0,
                         float color_hsv_min_s = 0.0, float color_hsv_min_v = 0.0) {

    if (std::isnan(intensity) || intensity == 0.0f)
        return rgb;

    if (std::abs(intensity) > 1.0f)
        intensity = std::copysign(1.0f, intensity);

    // RGB2HSV
    cv::Vec3f hsv = rgb2hsv(rgb);

    if (intensity > 0.0f) { // Brighten
        float di = 1.0f - intensity;
        if (hsv[1] != 0.0f)
            hsv[1] = di * hsv[1] + intensity * color_hsv_max_s;
        hsv[2] = di * hsv[2] + intensity * color_hsv_max_v;
    } else { // Darken
        float di = 1.0f + intensity;
        if (hsv[1] != 0.0f)
            hsv[1] = di * hsv[1] - intensity * color_hsv_min_s;
        hsv[2] = di * hsv[2] - intensity * color_hsv_min_v;
    }

    // 归一化0-1
    hsv[1] = std::max(0.0f, std::min(1.0f, hsv[1]));
    hsv[2] = std::max(0.0f, std::min(1.0f, hsv[2]));

    // HSV2RGB
    return hsv2rgb(hsv);
}
