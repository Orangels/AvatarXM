//
// Created by Orangels on 2023/9/6.
//
#include "task/avatarHandler.h"

#ifdef _WIN32
#else

#include <locale>
#include <codecvt>

#endif

avatarHandler::avatarHandler() {
    //int onnxruntime
    // 创建InferSession, 查询支持硬件设备
    // GPU Mode, 0 - gpu device id
    auto conf = config_A->getConfig();

    std::string onnxpath = conf["WENET"]["MODULE_PATH"].as<string>();
    std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());

    mEnv = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "we-encoder-onnx");
    mSession_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    if (conf["TEST"]["DEBUG"].as<bool>() == true) {
        std::cout << "onnxruntime inference try to use GPU Device" << std::endl;
    }
    //gpu
    OrtSessionOptionsAppendExecutionProvider_CUDA(mSession_options, 0);
#ifdef _WIN32
    Ort::Session session_(mEnv, modelPath.c_str(), mSession_options);
    mSession_wenet = new Ort::Session(*mEnv, modelPath.c_str(), mSession_options);
#else
    // 创建转换器
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::string CStr = converter.to_bytes(modelPath.c_str());
    mSession_wenet = new Ort::Session(*mEnv, CStr.c_str(), mSession_options);
#endif
}

avatarHandler::~avatarHandler() {
    mSession_options.release();
    mSession_wenet->release();
    mEnv->release();

    delete mSession_wenet;
    delete mEnv;
}

bool avatarHandler::run(int rows, int cols, int stride, float *feats) {
    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    size_t numInputNodes, numOutputNodes;

    numInputNodes = mSession_wenet->GetInputCount();
    numOutputNodes = mSession_wenet->GetOutputCount();

    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);

    // 获取输入信息
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = mSession_wenet->GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
    }

    // 获取输出信息
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = mSession_wenet->GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }

    // set input data and inference
    std::vector<int64_t> input_feats_node_dims = {1, rows, cols};
    size_t input_feats_tensor_size = 1 * rows * cols;

    auto feats_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_feats_tensor_ = Ort::Value::CreateTensor<float>(feats_info, feats, input_feats_tensor_size,
                                                                     input_feats_node_dims.data(), 3);

    std::vector<int64_t> input_length_node_dims = {1};
    size_t input_length_tensor_size = 1;
    std::vector<int32_t> input_length_tensor_values(input_length_tensor_size);
    for (unsigned int i = 0; i < input_length_tensor_size; i++)
        input_length_tensor_values[i] = rows;

    // create input tensor object from data values
    auto length_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_length_tensor_ = Ort::Value::CreateTensor<int32_t>(length_info, input_length_tensor_values.data(),
                                                                        input_length_tensor_size,
                                                                        input_length_node_dims.data(), 1);
    assert(input_length_tensor_.IsTensor());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_feats_tensor_));
    ort_inputs.push_back(std::move(input_length_tensor_));

    const std::array<const char *, 2> inputNames = {input_node_names[0].c_str(), input_node_names[1].c_str()};
    const std::array<const char *, 5> outNames = {output_node_names[0].c_str(), output_node_names[1].c_str(),
                                                  output_node_names[2].c_str(), output_node_names[3].c_str(),
                                                  output_node_names[4].c_str()};

    std::cout << "onnxruntime inference start" << std::endl;
    std::vector<Ort::Value> ort_outputs;

    try {
        ort_outputs = mSession_wenet->Run(Ort::RunOptions{nullptr}, inputNames.data(), ort_inputs.data(),
                                          ort_inputs.size(), outNames.data(), outNames.size());
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }

    std::cout << "onnxruntime inference end" << std::endl;
    // Get pointer to output tensor float values
    float *pdata = ort_outputs[0].GetTensorMutableData<float>();

    /*for (int i = 0; i < 100; i++) {
        std::cout << pdata[i] << std::endl;
    }*/

    const Ort::TensorTypeAndShapeInfo &shapeInfo = ort_outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> sp = shapeInfo.GetShape();
    std::cout << "onnxruntime out length: " << sp << std::endl;    // 1 * t_out * 512

//    std::vector<vector<float>> out_n_512;
//    int pdata_index;
//    for (int j = 0; j < sp[1]; j++) {
//        vector<float> encoder_512;
//        for (int k = 0; k < sp[2]; k++) {
//            pdata_index = j * 512 + k;
//            encoder_512.push_back(pdata[pdata_index]);
////            cout << pdata[pdata_index] << " ";
//        }
////        cout << endl;
//        out_n_512.push_back(encoder_512);
//    }
//
//    vector<vector<vector<float>>> out_mel;
//    torch::Tensor mel_tensor = torch::zeros({ (long)out_n_512.size() - 4, 5, 512 }, torch::kFloat32).contiguous();
//    int tensor_gen_start = getTimestamp();
//    cout << "tensor copy start " << endl;
//    for (int m = 0; m < out_n_512.size() - 4;m++) {
//        std::vector<vector<float>> out_5_512(out_n_512.begin() + m, out_n_512.begin() + m +5);
//        out_mel.push_back(out_5_512); // 58 5 512
//        for (int i = 0; i < out_5_512.size(); ++i) {
//            const float* src_ptr = out_5_512[i].data();
//            float* dst_ptr = mel_tensor[m][i].data_ptr<float>();
//            std::memcpy(dst_ptr, src_ptr, sizeof(float) * 512);
//        }
//    }
//
//    int mel_length = out_mel.size();
////    torch::Tensor mel_tensor = torch::from_blob(out_mel.data(), { mel_length, 5, 512 }).clone();
////    cout << "tensor copy start " << endl;
////    clock_t tensor_gen_start = clock();
//    int channels = out_mel[0].size();
//    int length = out_mel[0][0].size();
//    cout << "out_mel size : " <<mel_length << " " << channels << " " << length << endl;
////    torch::Tensor mel_tensor = torch::zeros({ mel_length, channels, length }, torch::kFloat32).contiguous();
//////
////    float* data_ptr = mel_tensor.data_ptr<float>();
////    std::memcpy(data_ptr, out_mel.data(), out_mel.size() * sizeof(std::vector<std::vector<float>>));
//    mel_tensor.unsqueeze_(1);
////
////    for (int i = 0; i < out_mel.size(); i++) {
////	    for (int j = 0; j < out_mel[i].size(); j++) {
////		    for (int k = 0; k < out_mel[i][j].size(); k++) {
////			    mel_tensor[i][0][j][k] = out_mel[i][j][k];
////		    }
////	    }
////    }
////    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
////    torch::Tensor mel_tensor = torch::from_blob(out_mel.data(), {mel_length,1, 5, 512}, opts).clone();
//    int tensor_gen_end = getTimestamp();
//    cout << "tensor copy end " << endl;
//    cout << "tenso copy cost : " << tensor_gen_end - tensor_gen_start << endl;
//    cout << mel_tensor.sizes() << endl;
//    cout << mel_tensor.dim() << endl;
//    cout << out_mel[0][0][0] << endl;
//    cout << "*************" << endl;
//    cout << mel_tensor[0][0][0][0].item<float>() << endl;
//    torch::Tensor mel_tensor_cuda = mel_tensor.to(torch::kCUDA);

    return true;
}
