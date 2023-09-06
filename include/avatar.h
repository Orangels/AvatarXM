#ifndef DISPATCH_H
#define DISPATCH_H

#include <iostream>

#include <opencv2/core.hpp>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <condition_variable>

#include "utils/singleton.h"
#include "utils/config_yaml.h"

using namespace std;

class Avatar{
public:
    Avatar();
    ~Avatar() = default;
    void run();
private:
    vector<queue<cv::Mat> *> mQueueCam;
    vector<queue<cv::Mat>>   mQueue_rtmp;

    vector<condition_variable *> mCon_not_full;
    vector<condition_variable *> mCon_not_empty;
    vector<condition_variable *> mCon_rtmp;
    vector<mutex *>              mConMutexCam;
    vector<mutex *>              mConMutexRTMP;
    vector<mutex *>              mRtmpMutex;


    void ProduceImage(int mode);

    void ConsumeImage(int mode);

    void ConsumeRTMPImage(int mode);

    void RPCServer();


    condition_variable vCon_not_full_0, vCon_not_full_1, vCon_not_full_2, vCon_not_full_3;
    condition_variable vCon_not_empty_0, vCon_not_empty_1, vCon_not_empty_2, vCon_not_empty_3;
    condition_variable vCon_rtmp_0, vCon_rtmp_1, vCon_rtmp_2, vCon_rtmp_3;
    mutex              vConMutexCam_0, vConMutexCam_1, vConMutexCam_2, vConMutexCam_3;
    mutex              vConMutexRTMP_0, vConMutexRTMP_1, vConMutexRTMP_2, vConMutexRTMP_3;
    mutex              vRtmpMutex_0, vRtmpMutex_1, vRtmpMutex_2, vRtmpMutex_3;
};
#endif