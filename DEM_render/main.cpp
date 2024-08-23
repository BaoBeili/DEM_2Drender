//
//  main.cpp
//  DEM_render
//
//  Created by huanghongli on 2024/7/17.
//

#include <iostream>
#include "gdal.h"
#include "gdal_priv.h"
#include "opencv2/opencv.hpp"
#include "grid_N.hpp"
#include <cmath>
#include "apply_colorMap.hpp"

// 定义常量
const double D2R = M_PI / 180.0; // 度到弧度的转换因子
const double one = 1.0;
const double DIST_M_PR_DEG = 111320.0;// WGS-84的每一度对应的米数


void saveColorImage(const cv::Mat& color_image, const std::string& outputFilename, const char* projection, double geotransform[6]) {
    // Initialize GDAL
    GDALAllRegister();

    int xSize = color_image.cols;
    int ySize = color_image.rows;
    
    // Create the output TIFF dataset
    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (driver == nullptr) {
        std::cerr << "GTiff driver not available." << std::endl;
        return;
    }

    GDALDataset *dataset = driver->Create(outputFilename.c_str(), xSize, ySize, 3, GDT_Byte, nullptr);
    if (dataset == nullptr) {
        std::cerr << "Failed to create output TIFF file." << std::endl;
        return;
    }

    // Set spatial reference (projection)
    if (projection != nullptr) {
        dataset->SetProjection(projection);
    }

    // Set geotransform parameters (affine transformation)
    if (geotransform != nullptr) {
        dataset->SetGeoTransform(geotransform);
    }

    // Write the RGB bands
    for (int bandIdx = 0; bandIdx < 3; ++bandIdx) {
        GDALRasterBand *band = dataset->GetRasterBand(bandIdx + 1);
        if (band == nullptr) {
            std::cerr << "Failed to get raster band." << std::endl;
            GDALClose(dataset);
            return;
        }
        
        cv::Mat channel(ySize, xSize, CV_8UC1);
        for (int i = 0; i < ySize; ++i) {
            for (int j = 0; j < xSize; ++j) {
                channel.at<uchar>(i, j) = color_image.at<cv::Vec3b>(i, j)[bandIdx];
            }
        }
        
        band->RasterIO(GF_Write, 0, 0, xSize, ySize, channel.data, xSize, ySize, GDT_Byte, 0, 0);
    }

    // Close the dataset
    GDALClose(dataset);
    
    // Check if the file exists
    if (std::filesystem::exists(outputFilename)) {
        std::cout << "Image successfully saved to " << outputFilename << std::endl;
    } else {
        std::cerr << "Failed to save image to " << outputFilename << std::endl;
    }
}



int main(int argc, const char * argv[]) {
    // 定义Ctrl结构体：
    // argument:储存我们所需要的参数信息
    // 参数包括了：
    // double azimuth入射角参数，设置默认为270度
    // N_mode N
    //    { mode:渲染格式Nt Ne N，设置为2，即Ne（高斯函数渲染）
    //      sigma:渲染系数，
    //      ambient:偏移系数，设置为0即可
    // }
    // CPTfiles：保存色卡文件路径
    
    Ctrl argument;// 参数
    argument.azimuth =270; // 入射角，默认为270度，可以修改
    argument.N.mode = 2; // 设置渲染格式，Nt,Ne,N，
    argument.a = 1 ; // 默认为1,强度缩放因子，
    argument.N.ambient = 0.0;
    argument.CPTfiles = "/Users/huanghongli/Documents/Code/Xcode_项目/DEM_render/colormapping/GMT/GMT_dem4.cpt";
    // 保证光线入射角在0到360度之间
    while(argument.azimuth>360)
    {
        argument.azimuth = argument.azimuth-360;
    }
    while(argument.azimuth<0)
    {
        argument.azimuth = argument.azimuth+360;
    }

    GDALAllRegister();
    // 要读取的 TIFF 文件路径
    std::string filename = "/Users/huanghongli/Documents/Code/Xcode_项目/二维渲染/zhibei.tif";// 接口
    // 打开 TIFF 文件
    GDALDataset *dataset = (GDALDataset *) GDALOpen(filename.c_str(), GA_ReadOnly);
    if (dataset == nullptr) {
        std::cerr << "无法打开文件：" << filename << std::endl;
    return 1;
    }
        
    // 初始化影像的信息
    int xSize = dataset->GetRasterXSize();
    int ySize = dataset->GetRasterYSize();
    GDALRasterBand *band = dataset->GetRasterBand(1);
    if (band == nullptr) {
        std::cerr << "无法获取波段" << std::endl;
        GDALClose(dataset);
        return 1;
    }
    
    // 获取网格间隔dx_grid 和 dy_grid
    // 在地理坐标系下时，dx_grid和dy_grid的单位为度，我们需要转换为米
    // 在投影坐标系下时，dx_grid和dy_grid的单位为米，我们不用转换。
    // STEP1: 获取网格间距:包括获取地理坐标系下和投影坐标系下
    double geotransform[6];
    if (dataset->GetGeoTransform(geotransform) != CE_None) {
        std::cerr << "Failed to get geotransform." << std::endl;
        GDALClose(dataset);
        return 1;
    }
    double dx_grid, dy_grid;
    double pixelSizeX = geotransform[1];
    double pixelSizeY = std::abs(geotransform[5]);
    const char* projection = dataset->GetProjectionRef();    // 查看坐标系统
    OGRSpatialReference oSRS;
    oSRS.importFromWkt(projection);
    bool isGeographic = oSRS.IsGeographic();// 判断是否为地理坐标系投影
    // 如果是地理坐标系需要重新计算出x和y方向上的大小偏差
    if(isGeographic)
    {
        double northLat = geotransform[3];// 获取纬度范围
        double southLat = geotransform[3] + geotransform[5] * dataset->GetRasterYSize();
        double midLat = (northLat + southLat) / 2.0;
        dx_grid = DIST_M_PR_DEG * pixelSizeX * cosd(midLat);
        dy_grid = DIST_M_PR_DEG * pixelSizeY;
    }
    else// 投影坐标系
    {
        dx_grid = pixelSizeX;
        dy_grid = pixelSizeY;
    }
    // 查看网格间隔
    std::cout<<dx_grid<<std::endl;
    std::cout<<dy_grid<<std::endl;
    
    // 初始化Grid和header
    GridHeader header;
    header.n_columns = xSize;
    header.n_rows = ySize;
    header.mx = xSize;

    // 储存高度信息，利用该信息后续计算出坡度坡向
    Grid grid;
    grid.header = &header;
    grid.data = new float[xSize * ySize];
    band->RasterIO(GF_Read, 0, 0, xSize, ySize, grid.data, xSize, ySize, GDT_Float32, 0, 0);
    
    // 储存高度信息，后续利用高度信息计算出初始的RGB值
    float *height = new float[xSize * ySize];
    band->RasterIO(GF_Read, 0, 0, xSize, ySize, height, xSize, ySize, GDT_Float32, 0, 0);
    
    // 由于原始的DEM中有NaN值和0值，NaN值的出现会影响最大高度和最小高度的计算
    // 计算出DEM最大的高度以及最小的高度，后续可以计算出RGB的值
    // 将数据中小于等于0的值设置为NaN，被认为是不合理高程值
    for (int i = 0; i < xSize * ySize; i++) {
        if (height[i] <= 0.0f) {
            height[i] = std::nanf("");
        }
    }
    std::vector<float> validHeights;
    for (int i = 0; i < xSize * ySize; i++) {
        if (!std::isnan(height[i])) {
            validHeights.push_back(height[i]);
        }
    }
    float minEle = *std::min_element(validHeights.begin(), validHeights.end());
    float maxEle = *std::max_element(validHeights.begin(), validHeights.end());
    
    std::vector<std::vector<float>> DEM_render_R(ySize, std::vector<float>(xSize));
    std::vector<std::vector<float>> DEM_render_G(ySize, std::vector<float>(xSize));
    std::vector<std::vector<float>> DEM_render_B(ySize, std::vector<float>(xSize));
    
    // 初始化设置
    argument.azimuth = D2R*argument.azimuth;// 将度转换为弧度
    double denom = 0.0;
    double rpi = 0.0;
    double sin_Az = sin(argument.azimuth);
    double x_factor = 0.0;
    double x_factor_set = one / (2.0 * dx_grid);
    x_factor_set = x_factor_set * sin_Az;
    double y_factor = one / (2.0 * dy_grid);
    double y_factor_set = one / (2.0 * dy_grid);
    y_factor = y_factor * cos(argument.azimuth);
    int p[4];// 计算某一像素上面下面左边右边的索引，以此获取周围高程，从而计算出梯度
    p[0] = 1;    p[1] = -1;    p[2] = header.mx;    p[3] = -header.mx;
    double min_gradient = DBL_MAX;
    double max_gradient = -DBL_MAX;
    double ave_gradient = 0.0;
    double n_used = 0;
    float color_hsv_max_s = 1.0f;
    float color_hsv_max_v = 1.0f;
    float color_hsv_min_s = 0.0f;
    float color_hsv_min_v = 0.0f;
    
    // 计算梯度
    for(int row =0;row<ySize;++row)
    {
        // 如果是地理坐标系则需要计算每像素的一个纬度信息
        if(isGeographic)
        {
            double lat = geotransform[3] - pixelSizeY*row;
            dx_grid = DIST_M_PR_DEG*pixelSizeX * cosd(lat);
            if (dx_grid > 0.0){
                x_factor = x_factor_set = one / (2.0 * dx_grid);
            }
        }
        // 投影坐标系
        else
        {
            x_factor = x_factor_set;
        }
        for(int col=0;col<xSize;++col)
        {
            // 计算索引位置
            int ij = row*xSize + col;
            bool bad = false;
            for (int n = 0; n < 4; ++n)
            {
                // 如果输入像素为NaN值，bad等于true，该点的梯度值为NaN
                if (ij + p[n] <0 || ij + p[n] > ySize*xSize ||gmt_M_is_fnan(height[ij + p[n]]))
                {
                    bad = true;
                }
                
            }
            if(bad)// 如果该点的上下左右四个元素有一个或者多个为NaN，则进入下一循环
            {
                grid.data[ij] = NAN;
                continue;
            }
            double dzdx = (height[ij+1] - height[ij-1])*x_factor;
            double dzdy = (height[ij-header.mx] - height[ij+header.mx]) * y_factor;
            double output = dzdx + dzdy;
            ave_gradient += output;
            min_gradient = MIN(min_gradient, output);
            max_gradient = MAX(max_gradient, output);
            grid.data[ij] = output;
            n_used++;// 可以使用的点的数目，以便于后续计算ave
        }
    }
    
    std::cout<<"n_used: "<<n_used<<std::endl;
    std::cout<<"x_factor: "<<x_factor<<std::endl;
    std::cout<<"y_factor: "<<y_factor<<std::endl;
    std::cout<<"min_gradient: "<<min_gradient<<std::endl;
    std::cout<<"max_gradient: "<<max_gradient<<std::endl;
    std::cout<<"min_Elevation: "<<minEle<<std::endl;
    std::cout<<"max_Elevation: "<<maxEle<<std::endl;
    
    // 利用梯度，计算出阴影的“强度”
    ave_gradient = ave_gradient/n_used;
    // mode=1:  ATAN transformation 反正切
    if (argument.N.mode == 1)
    {
        std::cout<<"反正切Nt"<<std::endl;
        for (int i=0;i<ySize;i++)
        {
            for(int j=0;j<xSize;j++)
            {
                if(!gmt_M_is_fnan(grid.data[i*xSize+j]))
                {
                    denom += pow(grid.data[i*xSize+j]-ave_gradient,2.0);
                }
            }
        }
        denom = sqrt ((n_used - 1) / denom);
        argument.N.sigma = 1.0/denom;
        rpi = 2*argument.a/M_PI;
        for (int i=0;i<ySize;i++)
        {
            for(int j=0;j<xSize;j++)
            {
                if(!gmt_M_is_fnan(grid.data[i*xSize+j]))
                {
                    grid.data[i*xSize+j] = rpi * atan((grid.data[i*xSize+j]-ave_gradient)*denom)+argument.N.ambient;
                }
            }
        }
        header.z_max = rpi * atan ((max_gradient - ave_gradient) * denom) + argument.N.ambient;
        header.z_min = rpi * atan ((min_gradient - ave_gradient) * denom) + argument.N.ambient;
    }
    //mode=2: Exp transformation 高斯变换
    else if(argument.N.mode == 2)
    {
        std::cout<<"高斯变换Ne"<<std::endl;
         #pragma omp parallel for reduction(+:argument.N.sigma)
        for (int i=0;i<ySize;i++){
            for(int j=0;j<xSize;j++){
                if(gmt_M_is_fnan(grid.data[i*xSize+j]))
                {
                    continue;
                }
                argument.N.sigma += fabs(grid.data[i*xSize+j]);
            }
        }
        argument.N.sigma = M_SQRT2*argument.N.sigma/n_used;
        denom = M_SQRT2/argument.N.sigma;

        for (int i=1;i<ySize-1;i++){
            for(int j=1;j<xSize-1;j++){
                if(gmt_M_is_fnan(grid.data[i*xSize+j]))
                    continue;
                if(grid.data[i*xSize+j]<ave_gradient){
                    grid.data[i*xSize+j] = (-argument.a * (1.0 - exp ( (grid.data[i*xSize+j] - ave_gradient) * denom)) + argument.N.ambient);
                }
                if(grid.data[i*xSize+j]>ave_gradient){
                    grid.data[i*xSize+j] = (argument.a * (1.0 - exp (-(grid.data[i*xSize+j] - ave_gradient) * denom)) + argument.N.ambient);
                }
            }
        }
        header.z_max = argument.a * (1.0 - exp (-(max_gradient - ave_gradient) * denom)) + argument.N.ambient;
        header.z_min = -argument.a * (1.0 - exp ( (min_gradient - ave_gradient) * denom)) + argument.N.ambient;
    }
    // mode = 0:线性变化
    else if(argument.N.mode == 0)
    {
        std::cout<<"线性变化"<<std::endl;
        if ((max_gradient - ave_gradient) > (ave_gradient - min_gradient))
            denom = argument.a / (max_gradient - ave_gradient);
        else
            denom = argument.a / (ave_gradient - min_gradient);
        for (int i=0;i<ySize;i++){
            for(int j=0;j<xSize;j++){
                if(!gmt_M_is_fnan(grid.data[i*xSize+j]))
                {
                    grid.data[i*xSize+j] = ((grid.data[i*xSize+j] - ave_gradient) * denom) + argument.N.ambient;
                }
            }
        }
        header.z_max = (max_gradient - ave_gradient) * denom + argument.N.ambient;
        header.z_min = (min_gradient - ave_gradient) * denom + argument.N.ambient;
    }
    
    // 读取 CPT 文件，组成ColorBar
    TColorMap colorMap;
    bool opencptFile = parserCptFile(argument.CPTfiles, colorMap);
    if(opencptFile){
        std::cout << colorMap << std::endl;
        std::cout<<"成功读取CPT文件！"<<std::endl;
    }else{
        std::cout<<"CPT文件读取失败！"<<std::endl;
    }
    // 获取 colormap.mItems 的值
    std::vector<TColorItem> items = colorMap.mItems;
    // 删除重复的RGB的值
    auto end = std::unique(items.begin(), items.end(), ColorItemComparer());
    items.erase(end, items.end());
    int nStops=items.size();
    // 获取色卡的值大小
    std::vector<ColorStop> gradient;
    for(int i = 0; i<nStops;++i){
        gradient.push_back(ColorStop(i/float(nStops), items[i].m_rgb[0], items[i].m_rgb[1], items[i].m_rgb[2]));
    }
    // 打印 gradient 中的所有 ColorStop
    for (const auto& color : gradient) {
        std::cout << color << std::endl;
    }
    
    // 遍历进行二维渲染得到颜色
    for (int i=1;i<ySize-1;i++){
        for(int j=1;j<xSize-1;j++){
            // 只有当像素值上所计算的grid.data值不为nan时，并且高程不为NaN时，成立
            if(!gmt_M_is_fnan(grid.data[i*xSize+j]))
            {
                // 计算初始的RGB值,利用 高度/（最大高度-最小高度）的值大小去计算比率，从而获取到初始的RGB
                auto color_RGB = linearGradient(gradient, (height[i*xSize+j]-minEle)/(maxEle-minEle));
                cv::Vec3f rgb(color_RGB[0], color_RGB[1], color_RGB[2]);
                
                // 输入mode2计算的强度信息，以及初始计算的RGB，计算出新的RGB。RGB归一化到0和1之间了。
                const cv::Vec3f& rgb_new = gmt_illuminate(rgb,grid.data[i*xSize+j], color_hsv_max_s, color_hsv_max_v, color_hsv_min_s, color_hsv_min_v);
                DEM_render_R[i][j]=rgb_new[0];//0-1;显示需要0-255，乘以255
                DEM_render_G[i][j]=rgb_new[1];//0-1;
                DEM_render_B[i][j]=rgb_new[2];//0-1;
            }
            else{
                DEM_render_R[i][j]=1;
                DEM_render_G[i][j]=1;
                DEM_render_B[i][j]=1;
            }

        }
    }
    
    // 利用OPenCV显示
    std::vector<cv::Mat> channels(3);
    channels[0] = cv::Mat(ySize, xSize, CV_8UC1); // Blue
    channels[1] = cv::Mat(ySize, xSize, CV_8UC1); // Green
    channels[2] = cv::Mat(ySize, xSize, CV_8UC1); // Red

    for (int i = 0; i < ySize; i++) {
        for (int j = 0; j < xSize; j++) {
            channels[0].at<uchar>(i, j) = static_cast<uchar>(DEM_render_R[i][j]*255.0f);
            channels[1].at<uchar>(i, j) = static_cast<uchar>(DEM_render_G[i][j]*255.0f);
            channels[2].at<uchar>(i, j) = static_cast<uchar>(DEM_render_B[i][j]*255.0f);
        }
    }
    cv::Mat color_image;
    cv::merge(channels, color_image);
    
    // 显示图像
//    cv::namedWindow("Color Image", cv::WINDOW_NORMAL);
//    cv::imshow("Color Image", color_image);
//    cv::waitKey(0);
    
    std::string outputFilename = "/Users/huanghongli/Documents/Code/Xcode_项目/DEM_render/output_color_image.tif";
    saveColorImage(color_image, outputFilename, projection, geotransform);
    
    delete[] height;
    delete[] grid.data;
    
    GDALClose(dataset);

}

