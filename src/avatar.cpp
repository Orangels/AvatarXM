#include "avatar.h"
#include "utils/extraAudioFeature.h"
#include <functional>

namespace {

    void avatarExtraAudioFeature(int rows, int cols, int stride, float *feats,
                                 Avatar *avatar) {
        avatar->audioCallback(rows, cols, stride, feats);
    }

    std::function<void(int, int, int, float *)> callBackTmp;

    void wrapperextraAudioFeature_File(int rows, int cols, int stride,
                                       float *feats) {
        callBackTmp(rows, cols, stride, feats);
    }
}// namespace

void Avatar::audioCallback(int rows, int cols, int stride, float *feats) {
    audioFeatRows   = rows;
    audioFeatCols   = cols;
    audioFeatStride = stride;
    audioFeats      = new float[rows * cols];
    std::memcpy(audioFeats, feats, sizeof(float) * rows * cols);
}

Avatar::Avatar() {
    callBackTmp = std::bind(&avatarExtraAudioFeature, std::placeholders::_1,
                            std::placeholders::_2, std::placeholders::_3,
                            std::placeholders::_4, this);

    mAvatarHandler = new avatarHandler();
    mWavToLipHandler = new wavToLip();
}

Avatar::~Avatar(){
    delete mAvatarHandler;
    delete mWavToLipHandler;
}

void Avatar::run(const char *audioPath) {

    extraAudioFeature_File(wrapperextraAudioFeature_File, audioPath);
    //    mAvatarHandler->run(audioFeatRows, audioFeatCols, audioFeatStride,
    //                        audioFeats);
    vector<cv::Mat> result;
//    result = mAvatarHandler->getWavToLipImgs(audioFeatRows, audioFeatCols, audioFeatStride,
//                                             audioFeats);
    result = mWavToLipHandler->getWavToLipImgs(audioFeatRows, audioFeatCols, audioFeatStride,
                                             audioFeats);
    cv::VideoWriter w1;
    w1.open("../result/test_result.mp4", cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 25, cv::Size(1080, 1920),
            true);
    int i = 0;
    for (cv::Mat frame : result){
        w1 << frame;
        cv::imwrite("../result/imgs/" + to_string(i++) + ".jpg", frame);
    }
    w1.release();

    delete[] audioFeats;
}

void Avatar::ProduceImage(int mode) { cout << "ProduceImage" << endl; }

void Avatar::ConsumeImage(int mode) { cout << "ConsumeImage" << endl; }

void Avatar::ConsumeRTMPImage(int mode) { cout << "ConsumeRTMPImage" << endl; }

void Avatar::RPCServer() { cout << "RPCServer" << endl; }