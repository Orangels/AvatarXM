//
// Created by ls on 2023/9/17.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <time.h>
#include <torch/script.h>
#include <torch/torch.h>

#define IMG_ORI "../c_file/pics/2222.jpg"
#define IMG_TENSOR "touch_mat_test.jpg"

using namespace cv;
using namespace std;
using namespace torch;

torch::Tensor mat2Tensor(cv::Mat frame)
{
    torch::Tensor res_tensor = torch::from_blob(frame.data,{1, 3, frame.rows, frame.cols}, torch::kFloat32);
    return std::move(res_tensor);
}

cv::Mat tensor2Mat(Tensor torch_tensor){
    cv::Mat mat(torch_tensor.size(2), torch_tensor.size(3), CV_8UC3);

    torch_tensor = torch_tensor.to(torch::kCPU);
    torch_tensor = torch_tensor.contiguous();

// 获取张量的指针和数据类型
    unsigned char* data_ptr = torch_tensor.data_ptr<unsigned char>();
    torch::ScalarType tensor_type = torch_tensor.scalar_type();

// 将数据复制到Mat对象中
    std::memcpy(mat.data, data_ptr, torch_tensor.size(2) * torch_tensor.size(3) * 3 * sizeof(unsigned char));

// 如果张量数据类型不是uint8，则需要进行类型转换
    if (tensor_type != torch::kUInt8) {
        mat.convertTo(mat, CV_8UC3);
    }
    return std::move(mat);
}

torch::Tensor matToTensor(const cv::Mat& image) {
    cv::Mat imageRGB;
    cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
    torch::Tensor tensor = torch::from_blob(imageRGB.data, {1, imageRGB.rows, imageRGB.cols, 3}, torch::kByte);
    tensor = tensor.permute({0, 3, 1, 2});
    tensor = tensor.to(torch::kFloat32).div_(255);
    return tensor;
}

// 将tensor转换为Mat对象
cv::Mat tensorToMat(const torch::Tensor& tensor) {
    torch::Tensor tensorRGB = tensor.permute({0, 2, 3, 1});
    tensorRGB = tensorRGB.mul_(255).clamp_(0, 255).to(torch::kUInt8);
    cv::Mat image(tensorRGB.size(1), tensorRGB.size(2), CV_8UC3, tensorRGB.data_ptr());
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    return image;
}

int main(){
    cv::Mat img_ori = cv::imread(IMG_ORI);
//    Tensor tensor_mat = mat2Tensor(img_ori);
//    cv::Mat img_tensor = tensor2Mat(tensor_mat);

    Tensor tensor_mat = matToTensor(img_ori);
    Mat mat_tensor = tensorToMat(tensor_mat);

    imwrite(IMG_TENSOR, mat_tensor);
}