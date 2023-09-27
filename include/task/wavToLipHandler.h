//
// Created by ls on 2023/9/27.
//

#ifndef AVATARXM_WAVTOLIPHANDLER_H
#define AVATARXM_WAVTOLIPHANDLER_H
#include <iostream>

#include <torch/torch.h>
#include <torch/script.h>
#include <onnxruntime_cxx_api.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils/singleton.h"
#include "utils/config_yaml.h"

class wavToLip {
public:
    wavToLip();

    ~wavToLip();

    bool run(int rows, int cols, int stride, float *feats);
    vector<cv::Mat> getWavToLipImgs(int rows, int cols, int stride, float *feats);
    int getSelectedTemplate() {
        return mSelectedTemplate;
    }

    void setSelectedTemplate(int num){
        mSelectedTemplate = num;
    }

private:

    bool preInfer();
    std::vector<int64_t> onnxinference(int rows, int cols, int stride, float *feats, float** pdata);

    //libTorch
    torch::jit::Module mWav2lip_model;
    torch::jit::Module mParsing_model;

    cv::FileStorage fs2;
    cv::VideoCapture mCapture;
    cv::VideoWriter mWriter;

    //onnxruntime
    Ort::Session *mSession_wenet;

    Ort::SessionOptions mSession_options;
    Ort::Env *mEnv;

    yamlConfig *config_A = Singleton<yamlConfig>::GetInstance("../cfg/avatarXM.yaml");

    int mSelectedTemplate = 0;
};
#endif //AVATARXM_WAVTOLIPHANDLER_H
