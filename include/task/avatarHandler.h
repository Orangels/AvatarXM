//
// Created by Orangels on 2023/9/6.
//

#ifndef AVATARXM_AVATARHANDLER_H
#define AVATARXM_AVATARHANDLER_H

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <onnxruntime_cxx_api.h>

#include "utils/singleton.h"
#include "utils/config_yaml.h"

class avatarHandler{
public:
    avatarHandler();
    ~avatarHandler()=default;
    bool infer_libtorch();
    bool infer_wenet();
    bool infer_yolov8();

private:

//    std::string wav2lip_model_pb;
    //libTorch
    torch::jit::Module * wav2lip_model;
    torch::jit::Module * parsing_model;
    //onnxruntime
    Ort::Session * session_wenet;
    Ort::Session * session_yolo;




    yamlConfig *config_A = Singleton<yamlConfig>::GetInstance("../cfg/config.yaml");
};

#endif //AVATARXM_AVATARHANDLER_H
