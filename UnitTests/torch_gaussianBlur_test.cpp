//
// Created by ls on 2023/9/15.
//
#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

torch::Tensor gaussianBlur(const torch::Tensor& input, float sigma) {
    // 创建高斯核
    int kernel_size = 5;
    int padding = kernel_size / 2;
    float variance = sigma * sigma;
    float coeff = 1.0f / (2.0f * M_PI * variance);
    torch::Tensor kernel = torch::empty({kernel_size, kernel_size});
    float* kernel_data = kernel.data_ptr<float>();
    float sum = 0.0f;

    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int x = i - padding;
            int y = j - padding;
            float value = coeff * std::exp(-(x * x + y * y) / (2.0f * variance));
            kernel_data[i * kernel_size + j] = value;
            sum += value;
        }
    }

    // 归一化高斯核
    kernel = kernel / sum;

    // 创建卷积层并加载高斯核
    torch::nn::Conv2d conv_layer(torch::nn::Conv2dOptions(1, 1, kernel_size).padding(kernel_size / 2));
    conv_layer->weight.data() = kernel.unsqueeze(0).unsqueeze(0);

    // 使用卷积层进行高斯模糊
    torch::Tensor blurred = conv_layer->forward(input);


    // 在输入张量上应用高斯模糊
//    torch::Tensor blurred = torch::conv2d(input.unsqueeze(0), kernel.unsqueeze(0), torch::Conv2dOptions().padding(padding));

//    return blurred.squeeze(0);
    return blurred;
}

int main() {
    // 读取图像
    cv::Mat image = cv::imread("/home/suimang/ls-dev/Demo/ffmpeg_test/cmake-build-remote_ffmpeg_test/Biden_test.jpg");

    // 转换为张量
    torch::Tensor tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte);
    tensor = tensor.permute({0, 3, 1, 2});  // 调整通道顺序

    // 转换为浮点型张量
    tensor = tensor.to(torch::kFloat32) / 255.0;

    // 高斯模糊
    torch::Tensor blurred = gaussianBlur(tensor, 1.0);

    // 将张量转换回图像
//    blurred = blurred * 255.0;
//    blurred = blurred.to(torch::kByte);
//    blurred = blurred.permute({0, 2, 3, 1}).squeeze(0);
//    cv::Mat blurred_image(image.rows, image.cols, CV_8UC3, blurred.data_ptr());

// 将PyTorch张量转换为cv::Mat
    cv::Mat blurred_image(blurred.size(2), blurred.size(3), CV_32FC1, blurred.data_ptr<float>());

    // 可选：将图像归一化到0-255范围
    cv::normalize(blurred_image, blurred_image, 0, 255, cv::NORM_MINMAX);

    // 显示模糊后的图像
    cv::imwrite("blurred_test.jpg", blurred_image);

    return 0;
}