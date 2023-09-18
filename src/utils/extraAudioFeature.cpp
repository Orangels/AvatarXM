//
// Created by ls on 2023/9/18.
//
#include "utils/extraAudioFeature.h"
#include "base/kaldi-common.h"
#include "feat/feature-fbank.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"
#include "matrix/kaldi-vector.h"

int extraAudioFeature_File(AudioFeatureCallback cb,const char* filename)
{
    try {
        using namespace kaldi;
        AudioFeatureCallback callback = cb;
        bool binary = true;
        Input wavfile(filename, &binary);
        WaveHolder waveholder;
        if (!waveholder.Read(wavfile.Stream())) {
            return -1;
        }

        FbankOptions mFbankOptions;
        mFbankOptions.frame_opts.samp_freq = 16000;
        mFbankOptions.frame_opts.frame_shift_ms = 10;
        mFbankOptions.frame_opts.frame_length_ms = 25;
        mFbankOptions.frame_opts.dither = 0.0;
        mFbankOptions.frame_opts.preemph_coeff = 0.970000029;
        mFbankOptions.frame_opts.remove_dc_offset = true;
        mFbankOptions.frame_opts.window_type = "povey";
        mFbankOptions.frame_opts.round_to_power_of_two = true;
        mFbankOptions.frame_opts.blackman_coeff = 0.419999987;
        mFbankOptions.frame_opts.snip_edges = true;
        mFbankOptions.frame_opts.allow_downsample = false;
        mFbankOptions.frame_opts.allow_upsample = false;
        mFbankOptions.frame_opts.max_feature_vectors = -1;

        mFbankOptions.mel_opts.num_bins = 80;
        mFbankOptions.mel_opts.low_freq = 20;
        mFbankOptions.mel_opts.high_freq = 0;
        mFbankOptions.mel_opts.vtln_low = 100;
        mFbankOptions.mel_opts.vtln_high = -500;
        mFbankOptions.mel_opts.debug_mel = false;
        mFbankOptions.mel_opts.htk_mode = false;

        mFbankOptions.use_energy = false;
        mFbankOptions.energy_floor = 0;
        mFbankOptions.raw_energy = true;
        mFbankOptions.htk_compat = false;
        mFbankOptions.use_log_fbank = true;
        mFbankOptions.use_power = true;

        Fbank fbank(mFbankOptions);

        const WaveData& wave_data = waveholder.Value();
        BaseFloat vtln_warp_local = 1.0;
        Matrix<BaseFloat> data = wave_data.Data();
        SubVector<BaseFloat> waveform(data, 0);
        int SampFreq = mFbankOptions.frame_opts.samp_freq;
        Matrix<BaseFloat> feats;
        fbank.ComputeFeatures(waveform, SampFreq, vtln_warp_local, &feats);
        if (callback)
        {
            callback(feats.NumRows(), feats.NumCols(), feats.Stride(), feats.Data());
        }
        return 0;
    }catch (const std::exception& e) {
        printf("%s",e.what());
        return -1;
    }
}
int extraAudioFeature_Buffer(AudioFeatureCallback cb,unsigned char* data_buffer,int buffersize)
{
    try {
        using namespace kaldi;
        AudioFeatureCallback callback = cb;
        FbankOptions mFbankOptions;
        mFbankOptions.frame_opts.samp_freq = 16000;
        mFbankOptions.frame_opts.frame_shift_ms = 10;
        mFbankOptions.frame_opts.frame_length_ms = 25;
        mFbankOptions.frame_opts.dither = 0.0;
        mFbankOptions.frame_opts.preemph_coeff = 0.970000029;
        mFbankOptions.frame_opts.remove_dc_offset = true;
        mFbankOptions.frame_opts.window_type = "povey";
        mFbankOptions.frame_opts.round_to_power_of_two = true;
        mFbankOptions.frame_opts.blackman_coeff = 0.419999987;
        mFbankOptions.frame_opts.snip_edges = true;
        mFbankOptions.frame_opts.allow_downsample = false;
        mFbankOptions.frame_opts.allow_upsample = false;
        mFbankOptions.frame_opts.max_feature_vectors = -1;

        mFbankOptions.mel_opts.num_bins = 80;
        mFbankOptions.mel_opts.low_freq = 20;
        mFbankOptions.mel_opts.high_freq = 0;
        mFbankOptions.mel_opts.vtln_low = 100;
        mFbankOptions.mel_opts.vtln_high = -500;
        mFbankOptions.mel_opts.debug_mel = false;
        mFbankOptions.mel_opts.htk_mode = false;

        mFbankOptions.use_energy = false;
        mFbankOptions.energy_floor = 0;
        mFbankOptions.raw_energy = true;
        mFbankOptions.htk_compat = false;
        mFbankOptions.use_log_fbank = true;
        mFbankOptions.use_power = true;

        Fbank fbank(mFbankOptions);

        //tts为两字节uint8_t, 需要转换成kaldi格式的浮点数

        uint8_t b1;
        uint8_t b2;
        uint16_t b;
        unsigned int check = 1 << 15;

        const int32_t size = (buffersize & 1) ? buffersize / 2 + 1 : buffersize / 2;
        float* wavedata = new float[size];

        for (int i = 0; i < size; i++) {
            int j = 2 * i;
            b1 = data_buffer[j];
            b2 = data_buffer[j + 1];
            b = (b2 << 8) | b1;
            int minus = (b & check) == check ? 1 : 0;
            if (minus) {
                b = ~b + 1;
                wavedata[i] = -b * (float)1.00;
            }
            else {
                wavedata[i] = b * (float)1.00;
            }
        }

        SubMatrix<BaseFloat> data(wavedata, 1, size, size);
        SubVector<BaseFloat> waveform(data, 0);
        int SampFreq = mFbankOptions.frame_opts.samp_freq;
        BaseFloat vtln_warp_local = 1.0;
        Matrix<BaseFloat> feats;
        fbank.ComputeFeatures(waveform, SampFreq, vtln_warp_local, &feats);
        if (callback)
        {
            callback(feats.NumRows(), feats.NumCols(), feats.Stride(), feats.Data());
        }
        return 0;
    }
    catch (const std::exception& e) {
        printf("%s", e.what());
        return 0;
    }
}