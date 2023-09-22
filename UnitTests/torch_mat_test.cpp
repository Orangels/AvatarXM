//
// Created by ls on 2023/9/17.
//
#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <torch/script.h>
#include <torch/torch.h>

#define IMG_ORI "../c_file/pics/2222.jpg"
#define IMG_TENSOR "touch_mat_test.jpg"
#define IMG_TENSOR_SINGLE "touch_mat_test_single.jpg"

using namespace cv;
using namespace std;
using namespace torch;

torch::Tensor mat2Tensor(cv::Mat frame) {
    torch::Tensor res_tensor = torch::from_blob(frame.data, {1, 3, frame.rows, frame.cols}, torch::kFloat32);
    return std::move(res_tensor);
}

cv::Mat tensor2Mat(Tensor torch_tensor) {
    cv::Mat mat(torch_tensor.size(2), torch_tensor.size(3), CV_8UC3);

    torch_tensor = torch_tensor.to(torch::kCPU);
    torch_tensor = torch_tensor.contiguous();

    // 获取张量的指针和数据类型
    unsigned char     *data_ptr   = torch_tensor.data_ptr<unsigned char>();
    torch::ScalarType tensor_type = torch_tensor.scalar_type();

    // 将数据复制到Mat对象中
    std::memcpy(mat.data, data_ptr, torch_tensor.size(2) * torch_tensor.size(3) * 3 * sizeof(unsigned char));

    // 如果张量数据类型不是uint8，则需要进行类型转换
    if (tensor_type != torch::kUInt8) { mat.convertTo(mat, CV_8UC3); }
    return std::move(mat);
}

torch::Tensor matToTensor(const cv::Mat &image) {
    cv::Mat imageRGB;
    cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGB);
    torch::Tensor tensor = torch::from_blob(imageRGB.data, {1, imageRGB.rows, imageRGB.cols, 3}, torch::kByte);
    tensor = tensor.permute({0, 3, 1, 2});
    tensor = tensor.to(torch::kFloat32).div_(255);
    return tensor;
}

// 将tensor转换为Mat对象
cv::Mat tensorToMat(const torch::Tensor &tensor) {
    cout << 1 << endl;
    torch::Tensor tensorRGB = tensor.permute({0, 2, 3, 1});
    cout << 2 << endl;
    tensorRGB = tensorRGB.mul_(255).clamp_(0, 255).to(torch::kUInt8);
    cout << 3 << endl;
    cv::Mat image(tensorRGB.size(1), tensorRGB.size(2), CV_8UC3);
    cout << 4 << endl;
    std::memcpy((void *) image.data, tensorRGB.data_ptr(), sizeof(torch::kU8) * 3 * 667 * 1000);
    cout << 5 << endl;
    //    cv::Mat image(tensorRGB.size(1), tensorRGB.size(2), CV_8UC3, tensorRGB.data_ptr());

    //    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

    return image;
}

cv::Mat tensorToMatNoCp(const torch::Tensor &tensor) {
    cout << 1 << endl;
    torch::Tensor tensorRGB = tensor.permute({0, 2, 3, 1});
    cout << 1 << endl;
    tensorRGB = tensorRGB.mul_(255).clamp_(0, 255).to(torch::kUInt8);
    cout << 1 << endl;
    //cv::Mat image(667, 1000, CV_8UC3);

    cv::Mat image(tensorRGB.size(1), tensorRGB.size(2), CV_8UC3, tensorRGB.data_ptr());
    //    cv::Mat image(tensorRGB.size(1), tensorRGB.size(2), CV_32FC3, tensorRGB.data_ptr());
    cout << 1 << endl;
    //    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cout << 1 << endl;
    return image;
}

//int main() {
//    cv::Mat img_ori       = cv::imread(IMG_ORI);
//    cv::Mat img_ori_clone = img_ori.clone();
//
//    //    torch::Tensor img_tensor = torch::from_blob(img_ori_clone.data, {img_ori_clone.rows, img_ori_clone.cols, 3}, torch::kFloat32);
//    //    //    Tensor tensor_mat = mat2Tensor(img_ori);
//    //    //    cv::Mat img_tensor = tensor2Mat(tensor_mat);
//    //
//    //    Tensor tensor_mat = matToTensor(img_ori);
//    //
//    //    torch::Tensor total_tensor = torch::empty({5, 3, 667, 1000}, torch::kFloat32);
//    //
//    //    torch::Tensor tensor_tmp = tensor_mat.clone().squeeze(0);
//    //
//    //    std::vector<torch::Tensor> imgs_batch = {
//    //            img_tensor.clone(),
//    //            img_tensor.clone(),
//    //            img_tensor.clone(),
//    //            img_tensor.clone(),
//    //            img_tensor.clone(),
//    //    };
//    //
//    //    torch::Tensor batchTensor = torch::stack(imgs_batch, 0);
//    //
//    //    batchTensor = batchTensor.permute({0, 3, 1 ,2});
//    //
//    //    cout << batchTensor.sizes() << endl;
//    //
//    //    //    cout << tensor_tmp.sizes() << endl;
//    //    //    for (int i = 0; i < 5; ++i) {
//    //    //        total_tensor[i] = tensor_tmp.clone();
//    //    //    }
//    //
//    //    //    Mat mat_tensor = tensorToMat(tensor_mat);
//    //    //    Mat mat_tensor = tensorToMat(total_tensor);
//    //    Mat mat_tensor_single_cp = tensorToMat(tensor_tmp.clone().unsqueeze_(0));
//    //    Mat mat_tensor_single    = tensorToMatNoCp(tensor_tmp.clone().unsqueeze_(0));
//    //
//    //    //    Mat mat_tensor_cp = tensorToMat(total_tensor.clone());
//    //    //    Mat mat_tensor    = tensorToMatNoCp(total_tensor.clone());
//    //
//    //    Mat mat_tensor_cp = tensorToMat(batchTensor.clone());
//    //    Mat mat_tensor    = tensorToMatNoCp(batchTensor.clone());
//    //
//    //    imwrite("../result/mat_tensor.jpg", mat_tensor);
//    //    imwrite("../result/mat_tensor_cp.jpg", mat_tensor_cp);
//    //
//    //    imwrite("../result/mat_tensor_single.jpg", mat_tensor_single);
//    //    imwrite("../result/mat_tensor_single_cp.jpg", mat_tensor_single_cp);
//    //
//    //    Mat mat_tensor_ori = tensorToMatNoCp(tensor_mat);
//    //    imwrite("../result/mat_tensor_ori.jpg", mat_tensor_ori);
//
//    int matrixSize = 3;
//
//    // 创建5个矩阵数据，并将其转换为Tensor对象
//    float matrix1[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//    float matrix2[9] = {10, 11, 12, 13, 14, 15, 16, 17, 18};
//    float matrix3[9] = {19, 20, 21, 22, 23, 24, 25, 26, 27};
//    float matrix4[9] = {28, 29, 30, 31, 32, 33, 34, 35, 36};
//    float matrix5[9] = {37, 38, 39, 40, 41, 42, 43, 44, 45};
//
//    torch::Tensor tensor1 = torch::from_blob(matrix1, {matrixSize, matrixSize});
//    torch::Tensor tensor2 = torch::from_blob(matrix2, {matrixSize, matrixSize});
//    torch::Tensor tensor3 = torch::from_blob(matrix3, {matrixSize, matrixSize});
//    torch::Tensor tensor4 = torch::from_blob(matrix4, {matrixSize, matrixSize});
//    torch::Tensor tensor5 = torch::from_blob(matrix5, {matrixSize, matrixSize});
//
//    // 将这些Tensor对象放入一个std::vector中
//    std::vector<torch::Tensor> tensorVector = {tensor1, tensor2, tensor3, tensor4, tensor5};
//
//    // 使用torch::stack函数将Tensor向量堆叠成一个新的Tensor对象
//    torch::Tensor batchTensor_f = torch::stack(tensorVector, 0);
//
//    // 打印batchTensor的形状
//    std::cout << "Batch Tensor Shape: " << batchTensor_f.sizes() << std::endl;
//    std::cout << batchTensor_f << endl;
//
//    float *frame_f = new float[img_ori_clone.total() * img_ori_clone.channels()];
//
//    // 将frame的数据拷贝到frame_f上
//    memcpy(frame_f, img_ori_clone.data, img_ori_clone.total() * img_ori_clone.channels() * sizeof(float));
//
//    cv::Mat newFrame(img_ori_clone.rows, img_ori_clone.cols, CV_32FC(img_ori_clone.channels()));
//    //memcpy(newFrame.data, frame_f, newFrame.total() * newFrame.channels() * sizeof(float));
//    for (int i = 0; i < newFrame.total() * newFrame.channels(); i++) {
//        newFrame.data[i] = frame_f[i];
//    }
//    cv::Mat newFrameConverted;
//    newFrame.convertTo(newFrameConverted, CV_8UC3);
////    newFrame.convertTo(newFrame, CV_8UC3);
//    // 保存新的cv::Mat对象为tmp.jpg到本地
//    cv::imwrite("../result/tmp.jpg", newFrameConverted);
//    cv::imwrite("../result/tmp_ori.jpg", img_ori_clone);
//
//    return 0;
//}

int main() {
    // 读取本地图像到cv::Mat对象
    cv::Mat originalImage = cv::imread(IMG_ORI, cv::IMREAD_COLOR);

    // 创建一个与原始图像大小相同的float数组
    float *imageData = new float[originalImage.total() * originalImage.channels()];
    cout << originalImage.total() << "   " << originalImage.elemSize() << endl;
    cout << originalImage.isContinuous() << endl;

    //    memcpy(imageData, (void *)originalImage.data, originalImage.total() * originalImage.channels() * sizeof(float));

    // 将原始图像数据拷贝到float数组中
    //    for (int i = 0; i < originalImage.total() * originalImage.channels(); i++) {
    //        imageData[i] = static_cast<float>(originalImage.data[i]);
    //    }
    //    int index = 0;
    //    for (int row = 0; row < originalImage.rows; row++) {
    //        for (int col = 0; col < originalImage.cols; col++) {
    //            cv::Vec3b pixel = originalImage.at<cv::Vec3b>(row, col);
    //            imageData[index++] = static_cast<float>(pixel[0]);
    //            imageData[index++] = static_cast<float>(pixel[1]);
    //            imageData[index++] = static_cast<float>(pixel[2]);
    //        }
    //    }
    cv::Mat reshapedImage = originalImage.reshape(1, 1);

    // 将图像数据拷贝到float数组中
    reshapedImage.copyTo(cv::Mat(1, originalImage.total() * originalImage.channels(), CV_32FC1, imageData));


    // 创建新的cv::Mat对象，并从float数组中复制数据
    //    cv::Mat newImage(originalImage.rows, originalImage.cols, CV_32FC3, imageData);
    //
    //    // 转换数据类型为合适的类型（CV_8UC3），以便保存为图像
    //    cv::Mat newImageConverted;
    //    //    newImage.convertTo(newImageConverted, CV_8UC3);
    //
    //    // 将新的cv::Mat对象保存为图像文件到本地
    //    cv::imwrite("../result/output.jpg", newImage);

    torch::Tensor tensor1 =
                          torch::from_blob(imageData,
                                           {originalImage.rows, originalImage.cols, originalImage.channels()},
                                           torch::kFloat32);
    torch::Tensor tensor2 = torch::from_blob(originalImage.clone().data,
                                             {originalImage.rows, originalImage.cols, originalImage.channels()},
                                             torch::kFloat32);

    //将这些Tensor对象放入一个std::vector中
    std::vector<torch::Tensor> tensorVector_1 = {tensor1.clone(), tensor1.clone(), tensor1.clone(), tensor1.clone(),
                                                 tensor1.clone()};
    std::vector<torch::Tensor> tensorVector_2 = {tensor2.clone(), tensor2.clone(), tensor2.clone(), tensor2.clone(),
                                                 tensor2.clone()};


    //    // 使用torch::stack函数将Tensor向量堆叠成一个新的Tensor对象
    //    torch::Tensor batchTensor_1 = torch::stack(tensorVector_1, 0);
    //    batchTensor_1 = batchTensor_1.permute({0, 3, 1, 2});
    //    cout << batchTensor_1.sizes() << endl;
    //    cv::Mat mat_result_1 = tensorToMat(batchTensor_1.clone());
    //    cv::Mat mat_result_2 = tensorToMatNoCp(batchTensor_1.clone());
    //
    //    imwrite("../result/1.jpg", mat_result_1);
    //    imwrite("../result/2.jpg", mat_result_2);
    //
    //
    //    torch::Tensor batchTensor_2 = torch::stack(tensorVector_2, 0);
    //    batchTensor_2 = batchTensor_2.permute({0, 3, 1, 2});
    //    cout << batchTensor_2.sizes() << endl;
    //    cv::Mat mat_result_3 = tensorToMat(batchTensor_2.clone());
    //    cv::Mat mat_result_4 = tensorToMatNoCp(batchTensor_2.clone());
    //
    //    imwrite("../result/3.jpg", mat_result_3);
    //    imwrite("../result/4.jpg", mat_result_4);


    cv::Mat batchMat;
    cv::Mat matList[] = {originalImage.clone(), originalImage.clone(), originalImage.clone()};
    //    cv::vconcat(originalImage, originalImage.clone(), batchMat);
    cv::vconcat(matList, 3, batchMat);

    cout << batchMat.rows << " -- " << batchMat.cols << " -- " << batchMat.channels() << endl;
    torch::Tensor batchTensor_3 = torch::from_blob(batchMat.data, {3, originalImage.rows, originalImage.cols, 3});
    batchTensor_3 = batchTensor_3.permute({0, 3, 1, 2}).div_(255.);

    cout << batchTensor_3.sizes() << endl;
    //    cv::Mat mat_result_5 = tensorToMat(batchTensor_3.clone());
    cout << "start tensor to mat" << endl;
    cv::Mat mat_result_6 = tensorToMatNoCp(batchTensor_3.clone());

    //    imwrite("../result/5.jpg", mat_result_5);
    imwrite("../result/6.jpg", mat_result_6);

    // 释放分配的内存
    delete[] imageData;
    return 0;
}