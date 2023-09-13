#include "avatar.h"


Avatar::Avatar(){
    mAvatarHandler = new avatarHandler();
}

void Avatar::run(){
    cout<<"Avatar start\n"<<endl;
    cout <<"test remote " << endl;
}

void Avatar::ProduceImage(int mode){
    cout << "ProduceImage" << endl;
}

void Avatar::ConsumeImage(int mode){
    cout << "ConsumeImage" << endl;
}

void Avatar::ConsumeRTMPImage(int mode){
    cout << "ConsumeRTMPImage" << endl;
}

void Avatar::RPCServer(){
    cout << "RPCServer" << endl;
}