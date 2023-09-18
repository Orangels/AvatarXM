//
// Created by ls on 2023/9/18.
//

#ifndef AVATARXM_EXTRAAUDIOFEATURE_H
#define AVATARXM_EXTRAAUDIOFEATURE_H

typedef void(* AudioFeatureCallback)(int rows, int cols, int stride, float* feats);
int  extraAudioFeature_File(AudioFeatureCallback cb,const char* filename);
int  extraAudioFeature_Buffer(AudioFeatureCallback cb,unsigned char* data_buffer, int buffersize);

#endif //AVATARXM_EXTRAAUDIOFEATURE_H
