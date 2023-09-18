//
// Created by Orangels on 2023/9/6.
//

#ifndef AVATARXM_AVATARHANDLER_H
#define AVATARXM_AVATARHANDLER_H

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils/singleton.h"
#include "utils/config_yaml.h"

class avatarHandler {
public:
    avatarHandler();

    ~avatarHandler();

    bool run(int rows, int cols, int stride, float *feats);

private:

    bool preInfer();

//    std::string wav2lip_model_pb;
    //libTorch
    torch::jit::Module wav2lip_model;
    torch::jit::Module parsing_model;

    cv::FileStorage fs2;
    cv::VideoCapture capture;

    //onnxruntime
    Ort::Session *mSession_wenet;

    Ort::SessionOptions mSession_options;
    Ort::Env *mEnv;


    yamlConfig *config_A = Singleton<yamlConfig>::GetInstance("../cfg/avatarXM.yaml");
};

#endif //AVATARXM_AVATARHANDLER_H
