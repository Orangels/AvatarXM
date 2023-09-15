#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <time.h>
#include <torch/script.h>
#include <torch/torch.h>
//extra audio feature
#include "base/kaldi-common.h"
#include "feat/feature-fbank.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"
#include "matrix/kaldi-vector.h"
//#include <ATen/core/List.h>

#ifdef _WIN32
#else
#include <locale>
#include <codecvt>
#endif
#define AUDIO_PATH "/home/suimang/ls-dev/Demo/test_audio/bd.wav"
typedef void(* AudioFeatureCallback)(int rows, int cols, int stride, float* feats);

using namespace cv;
using namespace std;
using namespace torch::indexing;

bool infer_libtorch() {
    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Predicting on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else {
        std::cout << "Predicting on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    std::cout << "CUDA： " << torch::cuda::is_available() << std::endl;
    std::cout << "CUDNN:  " << torch::cuda::cudnn_is_available() << std::endl;
    std::cout << "GPU(s): " << torch::cuda::device_count() << std::endl;

    int b = 24;

    std::string model_pb = "../c_file/checkpoints/wav2lip_c_cuda.pt";
    auto module = torch::jit::load(model_pb);
    module.to(at::kCUDA);
    std::cout << "loaded." << std::endl;


    //torch::Tensor img_batch_in = torch::ones({ 1, 6, 256, 256 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    //torch::Tensor mel_batch_in = torch::ones({ 1, 1, 5, 512 }, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    torch::NoGradGuard no_grad;
    //torch::Tensor img_batch_in = torch::ones({ 1, 6, 256, 256 }).to(device_type);
    torch::Tensor img_batch_in = torch::rand({ b, 6, 256, 256 }, torch::device(torch::kCUDA));
    //img_batch_in.to(torch::kCUDA);

    torch::Tensor mel_batch_in = torch::rand({ b, 1, 5, 512 }, torch::device(torch::kCUDA));
    //mel_batch_in.to(torch::kCUDA);

    // ls change
//    std::vector<torch::jit::IValue> inputs;

//    auto inputs = c10::impl::GenericList(c10::StringType::get());
    auto inputs_1 = c10::impl::GenericList(c10::TensorType::get());
    inputs_1.push_back(mel_batch_in);
    auto inputs_2 = c10::impl::GenericList(c10::TensorType::get());
    inputs_2.push_back(img_batch_in);


    cout << "**************" << endl;
    cout << "creat inputs tensor suc" << endl;
    auto output = module.forward({ mel_batch_in, img_batch_in }).toTuple();



//    torch::Tensor output = module.forward({ mel_batch_in, img_batch_in }).toTensor();
    //auto output = module.forward({ inputs }).toTensor();
//    auto outputs_pre = module.forward({ inputs_1, inputs_2 }).toTuple();
//    auto outputs_pre = module.forward({ inputs }).toTuple();
//    auto outputs_pre2 = module.forward({ inputs }).toTuple();
//    auto outputs_pre3 = module.forward({ inputs }).toTuple();
    std::cout << "pre infer." << std::endl;

    clock_t start = clock();
    for (int i = 0; i < 100; i++) {
        clock_t wav_start = clock();
        torch::Tensor img_batch_in = torch::rand({ b, 6, 256, 256 }, torch::device(torch::kCUDA));
        //img_batch_in.to(torch::kCUDA);

        torch::Tensor mel_batch_in = torch::rand({ b, 1, 5, 512 }, torch::device(torch::kCUDA));
        //mel_batch_in.to(torch::kCUDA);

        // ls change
        //std::vector<torch::jit::IValue> inputs;
//#ifdef _WIN32
//        std::vector<torch::jit::IValue> inputs;
//        inputs.push_back(mel_batch_in);
//        inputs.push_back(img_batch_in);
//        auto outputs_t = module.forward({ inputs }).toTuple();
//#else
//        auto inputs_1 = c10::impl::GenericList(c10::TensorType::get());
//        auto inputs_2 = c10::impl::GenericList(c10::TensorType::get());
//        inputs_1.push_back(mel_batch_in);
//        inputs_2.push_back(img_batch_in);
////        auto outputs_t = module.forward({ inputs_1, inputs_2 }).toTuple();
//        auto outputs_t = module.forward({ img_batch_in, mel_batch_in }).toTuple();
//#endif

//        inputs.push_back(mel_batch_in);
//        inputs.push_back(img_batch_in);
//        auto outputs_t = module.forward({ inputs }).toTuple();
        auto outputs_t = module.forward({ mel_batch_in, img_batch_in }).toTuple();
        clock_t wav_end = clock();
        cout << "----------wav time: " << wav_end - wav_start << endl;
    }
    clock_t end = clock();
    cout << "loop 1000 time: " << end - start << endl;
//    auto outputs = module.forward({ inputs }).toTuple();
    auto outputs = module.forward({ mel_batch_in, img_batch_in }).toTuple();
    torch::Tensor out1 = outputs->elements()[0].toTensor();
    torch::Tensor out2 = outputs->elements()[1].toTensor();



    std::cout << out1.sizes() << std::endl;
    std::cout << out2.sizes() << std::endl;


    std::string model_pb2 = "../c_file/checkpoints/parsing_c_cuda.pt";
    auto module2 = torch::jit::load(model_pb2);
    module2.to(at::kCUDA);
    std::cout << "loaded." << std::endl;
    //torch::NoGradGuard no_grad;
    torch::Tensor batch_in = torch::ones({ b, 3, 384, 384 }, torch::device(torch::kCUDA));
    clock_t start1 = clock();
    for (int i = 0; i < 100; i++) {
        clock_t parsing_start = clock();
        torch::Tensor batch_in = torch::ones({ b, 3, 384, 384 }, torch::device(torch::kCUDA));
        auto outputs2 = module2.forward({ batch_in }).toTuple();
        clock_t parsing_end = clock();
        cout << "----------parsing time: " << parsing_end - parsing_start << endl;
    }
    clock_t end1 = clock();
    cout << "loop 1000 time: " << end1 - start1 << endl;
    auto outputs2 = module2.forward({ batch_in }).toTuple();
    torch::Tensor out12 = outputs2->elements()[0].toTensor();
    torch::Tensor out22 = outputs2->elements()[1].toTensor();
    torch::Tensor out32 = outputs2->elements()[1].toTensor();


    std::cout << out12.sizes() << std::endl;
    std::cout << out22.sizes() << std::endl;
    std::cout << out32.sizes() << std::endl;

    return true;
}


bool infer_wenet() {
    //Mat img = imread("D:\\face\\c\\AvatarXM\\file\\pics\\1111.jpg");
    //namedWindow("test opencv");
    //imshow("test opencv", img);
    //waitKey(6000);


    // 创建InferSession, 查询支持硬件设备
    // GPU Mode, 0 - gpu device id
    std::string onnxpath = "../c_file/checkpoints/encoder.onnx";
    std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
    Ort::SessionOptions session_options;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "we-encoder-onnx");

    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    std::cout << "onnxruntime inference try to use GPU Device" << std::endl;
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

#ifdef _WIN32
    Ort::Session session_(env, modelPath.c_str(), session_options);
#else
    // 创建转换器
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::string CStr = converter.to_bytes(modelPath.c_str());
    Ort::Session session_(env, CStr.c_str(), session_options);

#endif
//    Ort::Session session_(env, modelPath.c_str(), session_options);



    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;

    size_t numInputNodes = session_.GetInputCount();
    size_t numOutputNodes = session_.GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);



    // 获取输入信息
    int input_w = 0;
    int input_h = 0;
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = session_.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[2];	// 80
        input_h = input_dims[1];	// T_in
        std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << std::endl;
    }


    // 获取输出信息
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[1];		//
    output_w = output_dims[2];		//
    std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = session_.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }
    std::cout << "input: " << input_node_names[0] << " " << input_node_names[1] << std::endl;
    std::cout << "output: " << output_node_names[0] << " " << output_node_names[1] << " " << output_node_names[2] << " " << output_node_names[3] << " " << output_node_names[4] << std::endl;


    // set input data and inference
    std::vector<int64_t> input_feats_node_dims = { 1, 134, 80 };
    size_t input_feats_tensor_size = 1 * 134 * 80;
    std::vector<float> input_feats_tensor_values(input_feats_tensor_size);
    for (unsigned int i = 0; i < input_feats_tensor_size; i++)
        input_feats_tensor_values[i] = 1.0;

    auto feats_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_feats_tensor_ = Ort::Value::CreateTensor<float>(feats_info, input_feats_tensor_values.data(), input_feats_tensor_size, input_feats_node_dims.data(), 3);


    std::vector<int64_t> input_length_node_dims = { 1 };
    size_t input_length_tensor_size = 1;
    std::vector<int32_t> input_length_tensor_values(input_length_tensor_size);
    for (unsigned int i = 0; i < input_length_tensor_size; i++)
        input_length_tensor_values[i] = 134;
    // create input tensor object from data values
    auto length_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_length_tensor_ = Ort::Value::CreateTensor<int32_t>(length_info, input_length_tensor_values.data(), input_length_tensor_size, input_length_node_dims.data(), 1);
    assert(input_length_tensor_.IsTensor());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_feats_tensor_));
    ort_inputs.push_back(std::move(input_length_tensor_));




    const std::array<const char*, 2> inputNames = { input_node_names[0].c_str(), input_node_names[1].c_str() };
    const std::array<const char*, 5> outNames = { output_node_names[0].c_str(), output_node_names[1].c_str(), output_node_names[2].c_str(), output_node_names[3].c_str(), output_node_names[4].c_str() };


    //const std::array<const char*, 2> inputNames = { "speech", "speech_lengths" };
    //const std::array<const char*, 5> outNames = { "encoder_out", "encoder_out_lens", "ctc_log_probs", "beam_log_probs", "beam_log_probs_idx" };



    std::cout << "onnxruntime inference start" << std::endl;
    std::vector<Ort::Value> ort_outputs;

    //ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), ort_inputs.data(), ort_inputs.size(), outNames.data(), outNames.size());
    //// Get pointer to output tensor float values
    //float* floatarr = ort_outputs[0].GetTensorMutableData<float>();

    try {
        ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), ort_inputs.data(), ort_inputs.size(), outNames.data(), outNames.size());
        /*for (int i = 0; i < 100; i++) {
            clock_t ort_run_start = clock();
            ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), ort_inputs.data(), ort_inputs.size(), outNames.data(), outNames.size());
            clock_t ort_run_end = clock();
            cout << "----------cap time: " << ort_run_end - ort_run_start << endl;
        }*/

    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }

    std::cout << "onnxruntime inference end" << std::endl;
    // Get pointer to output tensor float values
    float* pdata = ort_outputs[0].GetTensorMutableData<float>();


    /*for (int i = 0; i < 100; i++) {
        std::cout << pdata[i] << std::endl;
    }*/

    const Ort::TensorTypeAndShapeInfo& shapeInfo = ort_outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> sp = shapeInfo.GetShape();
    std::cout << "onnxruntime out length: " << sp << std::endl;	// 1 * t_out * 512


    std::vector<vector<float>> out_n_512;
    int pdata_index;
    for (int j = 0; j < sp[1]; j++) {
        vector<float> encoder_512;
        for (int k = 0; k < sp[2]; k++) {
            pdata_index = j * 512 + k;
            encoder_512.push_back(pdata[pdata_index]);
        }
        out_n_512.push_back(encoder_512);
    }

    vector<vector<vector<float>>> out_mel;


    for (int m = 0; m < out_n_512.size() - 4;m++) {
        std::vector<vector<float>> out_5_512(out_n_512.begin() + m, out_n_512.begin() + m +5);
        out_mel.push_back(out_5_512);
    }

    int mel_length = out_mel.size();
    torch::Tensor mel_tensor = torch::from_blob(out_mel.data(), {mel_length,1, 5, 512});
    torch::Tensor mel_tensor_cuda = mel_tensor.to(torch::kCUDA);
    std::cout << "tensor size:  "<< mel_tensor.sizes() << std::endl;	// mel_length * 1 * 5 * 512
    //auto mel_tensor_in = mel_tensor.unsqueeze(1);
    //std::cout << "tensor size:  " << mel_tensor_in.sizes() << std::endl;// mel_length * 1 * 5 * 512
    //torch::Tensor mel_tensor = torch::from_blob(pdata, (-1, 512), torch::device(torch::kCUDA));

    cv::FileStorage fs2("../c_file/template/2023-04-21_22-06-37_24422621726989746_changed_head_fusion_vid.yml", cv::FileStorage::READ);
    cv::Mat face_det_boxes;
    fs2["face_det_boxes"] >> face_det_boxes;
    int frameFps = (int)fs2["fps"];
    int frameHeight = (int)fs2["frame_h"];
    int frameWidth = (int)fs2["frame_w"];

    // 构建face信息
    VideoCapture capture("../c_file/template/2023-04-21_22-06-37_24422621726989746_changed_head_fusion_vid.mp4");
    cout << "Cap open " << capture.isOpened() << endl;
    if (!capture.isOpened())
    {
        std::cout << "please check camera !!!" << std::endl;
        return false;
    }

    vector<Mat> frames_batch;
    vector<Mat> faces_batch;
    cv::Mat frame;
    cout << "mel_length " << mel_length << endl;
//    torch::Tensor frames_3_tensor = torch::zeros({ mel_length, 256, 256, 3 }, torch::kFloat32);
    torch::Tensor frames_3_tensor = torch::empty({ mel_length, 256, 256,3 }, torch::kByte);
    for (int n = 0; n < out_mel.size(); n++) {
        capture >> frame;
        frames_batch.push_back(frame);

        //拷贝构造函数2---只拷贝感兴趣的区域----由Rect对象定义
        //rect左上角（100，100），宽高均为200，（x,y,width,height)
        int y1 = face_det_boxes.at<int>(n, 0);
        int y2 = face_det_boxes.at<int>(n, 1);
        int x1 = face_det_boxes.at<int>(n, 2);
        int x2 = face_det_boxes.at<int>(n, 3);
        //cv::Rect rect(x1, y1, x2 - x1, y2 - y1);
        cv::Mat frame_face(frame, cv::Rect(x1, y1, x2 - x1, y2 - y1));
        cv::resize(frame_face, frame_face, cv::Size(256, 256));
        //frame_face.convertTo(frame_face, CV_32FC3, 1.0 / 255.0);

//        faces_batch.push_back(frame_face);
        std::memcpy(frames_3_tensor[n].data_ptr(), frame_face.data, frame_face.total() * frame_face.elemSize());
    }
    frames_3_tensor = frames_3_tensor.to(torch::kFloat32).div_(255).to(torch::kCUDA);

//    for (int i = 0; i < mel_length; i++) {
//        cv::Mat face = faces_batch[i];
//        torch::Tensor tensor = torch::from_blob(face.data, { 256, 256, 3 }, torch::kFloat32);
//        //tensor = tensor.permute({ 2, 0, 1 }).to(torch::kFloat32) / 255.0;
//        cout << "!!!!!!" << endl;
//        cout << tensor.dim() << endl;
//        cout << tensor.sizes() << endl;
//        cout << tensor.unsqueeze(0).dim() << endl;
//        cout << tensor.unsqueeze(0).sizes() << endl;
//        cout << frames_3_tensor[i].dim() << endl;
//        cout << frames_3_tensor[i].sizes() << endl;
//
//        try {
//            frames_3_tensor[i] = tensor;
//        }
//        catch (const c10::Error& e) {
//            std::cout << "catch_error: " << e.msg() << std::endl;
//        };
//
//    }

    cout << "frames_3_tensor done" << endl;
    //torch::Tensor frames_3_tensor_u8 = torch::from_blob(faces_batch.data(), ((int)out_mel.size(), 256, 256, 3), torch::dtype(torch::kFloat32));
    //auto ffff_c = frames_3_tensor_u8.clone();


    //torch::Tensor frames_3_tensor_u8_cuda = torch::from_blob(faces_batch.data(), ((int)out_mel.size(), 256, 256, 3), torch::dtype(torch::kFloat32).device(torch::kCUDA));
    //auto ffff_c_cuda = frames_3_tensor_u8_cuda.clone();


    //torch::Tensor frames_3_tensor_u8 = torch::from_blob(faces_batch.data(), ((int)out_mel.size(), 256, 256, 3), torch::dtype(torch::kFloat32).device(torch::kCUDA));
    //torch::Tensor frames_3_tensor = torch::from_blob(faces_batch.data(), ((int)out_mel.size(), 256, 256, 3 ),torch::device(torch::kCUDA)).to(torch::kFloat32);
//    torch::Tensor frames_3_tensor = torch::from_blob(faces_batch.data(), ((int)out_mel.size(), 256, 256, 3 )).to(torch::kFloat32);
    torch::Tensor frames_3_tensor_mask = frames_3_tensor.clone();
    //torch::Tensor frames_3_tensor_cuda = frames_3_tensor.to(torch::kCUDA);
    //torch::Tensor frames_3_tensor_mask_cuda = frames_3_tensor_mask.to(torch::kCUDA);

    frames_3_tensor_mask.index_put_({Slice(), Slice(128), Slice(), Slice() }, 0);
    torch::Tensor frames_6_tensor_cuda = torch::cat({ frames_3_tensor, frames_3_tensor_mask }, 3).permute({0, 3, 1, 2}).to(torch::kCUDA);

    std::string model_pb = "../c_file/checkpoints/wav2lip_c_cuda.pt";
    auto module = torch::jit::load(model_pb);
    module.to(at::kCUDA);
    std::cout << "loaded." << std::endl;


    // ls change
//    std::vector<torch::jit::IValue> inputs;
//    auto inputs = c10::impl::GenericList(c10::StringType::get());
//    inputs.push_back(mel_tensor_cuda);
//    inputs.push_back(frames_6_tensor_cuda);

//    auto outputs = module.forward({ inputs }).toTuple();
    auto outputs = module.forward({ mel_tensor_cuda, frames_6_tensor_cuda }).toTuple();
    torch::Tensor out1 = outputs->elements()[0].toTensor();
    torch::Tensor out2 = outputs->elements()[1].toTensor();

    std::cout << "pre infer." << std::endl;


    session_options.release();
    session_.release();
    capture.release();

    return true;
}


std::vector<std::string> readClassNames()
{
    std::string labels_txt_file = "../c_file/checkpoints/classes.txt";
    std::vector<std::string> classNames;

    std::ifstream fp(labels_txt_file);
    if (!fp.is_open())
    {
        printf("could not open file...\n");
        exit(-1);
    }
    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back(name);
    }
    fp.close();
    return classNames;
}

bool infer_yolov8() {
    std::vector<std::string> labels = readClassNames();
    cv::Mat frame = cv::imread("../c_file/pics/2222.jpg");
    int ih = frame.rows;
    int iw = frame.cols;

    // 创建InferSession, 查询支持硬件设备
    // GPU Mode, 0 - gpu device id
    std::string onnxpath = "../c_file/checkpoints/yolov8n.onnx";
    std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
    Ort::SessionOptions session_options;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-onnx");

    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    std::cout << "onnxruntime inference try to use GPU Device" << std::endl;
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
#ifdef _WIN32
    Ort::Session session_(env, modelPath.c_str(), session_options);
#else
    // 创建转换器
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::string CStr = converter.to_bytes(modelPath.c_str());
    Ort::Session session_(env, CStr.c_str(), session_options);

#endif
//    Ort::Session session_(env, modelPath.c_str(), session_options);

    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;

    size_t numInputNodes = session_.GetInputCount();
    size_t numOutputNodes = session_.GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);

    // 获取输入信息
    int input_w = 0;
    int input_h = 0;
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = session_.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[3];
        input_h = input_dims[2];
        std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
    }

    // 获取输出信息
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[1]; // 84
    output_w = output_dims[2]; // 8400
    std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = session_.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }
    std::cout << "input: " << input_node_names[0] << " output: " << output_node_names[0] << std::endl;

    // format frame
    int64 start = cv::getTickCount();
    int w = frame.cols;
    int h = frame.rows;
    int _max = std::max(h, w);
    cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
    cv::Rect roi(0, 0, w, h);
    frame.copyTo(image(roi));

    // fix bug, boxes consistence!
    float x_factor = image.cols / static_cast<float>(input_w);
    float y_factor = image.rows / static_cast<float>(input_h);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
    size_t tpixels = input_h * input_w * 3;
    std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

    // set input data and inference
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
    const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
    const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };
    std::vector<Ort::Value> ort_outputs;
    try {
        ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, 1, outNames.data(), outNames.size());
    }
    catch (std::exception e) {
        std::cout << e.what() << std::endl;
    }



    // output data
    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
    cv::Mat det_output = dout.t(); // 8400x84

    // post-process
    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    for (int i = 0; i < det_output.rows; i++) {
        cv::Mat classes_scores = det_output.row(i).colRange(4, 84);
        cv::Point classIdPoint;
        double score;
        minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

        // 置信度 0～1之间
        if (score > 0.25)
        {
            float cx = det_output.at<float>(i, 0);
            float cy = det_output.at<float>(i, 1);
            float ow = det_output.at<float>(i, 2);
            float oh = det_output.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
            int width = static_cast<int>(ow * x_factor);
            int height = static_cast<int>(oh * y_factor);
            cv::Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;

            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(score);
        }
    }

    // NMS
    std::vector<int> indexes;
    cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
    for (size_t i = 0; i < indexes.size(); i++) {
        int index = indexes[i];
        int idx = classIds[index];
        cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
        cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
                      cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
        putText(frame, labels[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
//        cv::imshow("YOLOv8+ONNXRUNTIME 对象检测演示", frame);
    }

    // 计算FPS render it
    float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
    putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
//    cv::imshow("YOLOv8+ONNXRUNTIME 对象检测演示", frame);
//    cv::waitKey(0);
    cv::imwrite("yolo_result.jpg", frame);

    session_options.release();
    session_.release();
    return true;

}


bool test_yml() {
    cv::FileStorage fs2("../c_file/template/2023-04-21_22-06-37_24422621726989746_changed_head_fusion_vid.yml", cv::FileStorage::READ);
    cv::Mat face_det_boxes;
    fs2["face_det_boxes"] >> face_det_boxes;
    int frameFps = (int)fs2["fps"];
    int frameHeight = (int)fs2["frame_h"];
    int frameWidth = (int)fs2["frame_w"];

    std::cout << "boxes size:  " << face_det_boxes.size() << std::endl;
    for (int i = 0; i < face_det_boxes.rows; i++) {
        std::cout << "boxes index_"<< i <<"::  " << face_det_boxes.at<int>(i, 0) << " " << face_det_boxes.at<int>(i, 1) << " " << face_det_boxes.at<int>(i, 2) << " " << face_det_boxes.at<int>(i, 3) << std::endl;
    }
    std::cout << "frameFps:  " << frameFps << std::endl;
    std::cout << "frameHeight:  " << frameHeight << std::endl;
    std::cout << "frameWidth:  " << frameWidth << std::endl;

    return true;


}

void test_cv_vid() {
    VideoCapture capture("../c_file/template/2023-04-21_22-06-37_24422621726989746_changed_head_fusion_vid.mp4");
    if (!capture.isOpened())
    {
        std::cout << "please check camera !!!" << std::endl;
        return;
    }

    int frames = capture.get(CAP_PROP_FRAME_COUNT);//获取视频针数目(一帧就是一张图片)
    double fps = capture.get(CAP_PROP_FPS);//获取每针视频的频率
    // 获取帧的视频宽度，视频高度
    Size size = Size(capture.get(CAP_PROP_FRAME_WIDTH), capture.get(CAP_PROP_FRAME_HEIGHT));
    cout << frames << endl;
    cout << fps << endl;
    cout << size << endl;
    // 创建视频中每张图片对象
    Mat frame;
    namedWindow("video-demo", WINDOW_AUTOSIZE);
    // 循环显示视频中的每张图片
    for (;;)
    {
        //将视频转给每一张张图进行处理
        capture >> frame;
        //省略对图片的处理
        //视频播放完退出
        if (frame.empty())break;
        imshow("video-demo", frame);
        //在视频播放期间按键退出
        if (waitKey(33) >= 0) break;
    }
    //释放
}

bool wav2lip(int rows, int cols, int stride, float* feats) {
    // 创建InferSession, 查询支持硬件设备
    // GPU Mode, 0 - gpu device id
    std::string onnxpath = "../c_file/checkpoints/encoder.onnx";
    std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
    Ort::SessionOptions session_options;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "we-encoder-onnx");

    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    std::cout << "onnxruntime inference try to use GPU Device" << std::endl;
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

#ifdef _WIN32
    Ort::Session session_(env, modelPath.c_str(), session_options);
#else
    // 创建转换器
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::string CStr = converter.to_bytes(modelPath.c_str());
    Ort::Session session_(env, CStr.c_str(), session_options);

#endif
//    Ort::Session session_(env, modelPath.c_str(), session_options);



    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;

    size_t numInputNodes = session_.GetInputCount();
    size_t numOutputNodes = session_.GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    input_node_names.reserve(numInputNodes);



    // 获取输入信息
    int input_w = 0;
    int input_h = 0;
    for (int i = 0; i < numInputNodes; i++) {
        auto input_name = session_.GetInputNameAllocated(i, allocator);
        input_node_names.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_w = input_dims[2];	// 80
        input_h = input_dims[1];	// T_in
        std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << std::endl;
    }


    // 获取输出信息
    int output_h = 0;
    int output_w = 0;
    Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_h = output_dims[1];		//
    output_w = output_dims[2];		//
    std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
    for (int i = 0; i < numOutputNodes; i++) {
        auto out_name = session_.GetOutputNameAllocated(i, allocator);
        output_node_names.push_back(out_name.get());
    }
    std::cout << "input: " << input_node_names[0] << " " << input_node_names[1] << std::endl;
    std::cout << "output: " << output_node_names[0] << " " << output_node_names[1] << " " << output_node_names[2] << " " << output_node_names[3] << " " << output_node_names[4] << std::endl;


    // set input data and inference
    std::vector<int64_t> input_feats_node_dims = { 1, rows, cols };
    size_t input_feats_tensor_size = 1 * rows * cols;

    auto feats_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_feats_tensor_ = Ort::Value::CreateTensor<float>(feats_info, feats, input_feats_tensor_size, input_feats_node_dims.data(), 3);


    std::vector<int64_t> input_length_node_dims = { 1 };
    size_t input_length_tensor_size = 1;
    std::vector<int32_t> input_length_tensor_values(input_length_tensor_size);
    for (unsigned int i = 0; i < input_length_tensor_size; i++)
        input_length_tensor_values[i] = rows;
    // create input tensor object from data values
    auto length_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_length_tensor_ = Ort::Value::CreateTensor<int32_t>(length_info, input_length_tensor_values.data(), input_length_tensor_size, input_length_node_dims.data(), 1);
    assert(input_length_tensor_.IsTensor());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_feats_tensor_));
    ort_inputs.push_back(std::move(input_length_tensor_));

    const std::array<const char*, 2> inputNames = { input_node_names[0].c_str(), input_node_names[1].c_str() };
    const std::array<const char*, 5> outNames = { output_node_names[0].c_str(), output_node_names[1].c_str(), output_node_names[2].c_str(), output_node_names[3].c_str(), output_node_names[4].c_str() };


    //const std::array<const char*, 2> inputNames = { "speech", "speech_lengths" };
    //const std::array<const char*, 5> outNames = { "encoder_out", "encoder_out_lens", "ctc_log_probs", "beam_log_probs", "beam_log_probs_idx" };

    std::cout << "onnxruntime inference start" << std::endl;
    std::vector<Ort::Value> ort_outputs;

    //ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), ort_inputs.data(), ort_inputs.size(), outNames.data(), outNames.size());
    //// Get pointer to output tensor float values
    //float* floatarr = ort_outputs[0].GetTensorMutableData<float>();

//    try {
        ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), ort_inputs.data(), ort_inputs.size(), outNames.data(), outNames.size());
        /*for (int i = 0; i < 100; i++) {
            clock_t ort_run_start = clock();
            ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), ort_inputs.data(), ort_inputs.size(), outNames.data(), outNames.size());
            clock_t ort_run_end = clock();
            cout << "----------cap time: " << ort_run_end - ort_run_start << endl;
        }*/

//    }
//    catch (std::exception e) {
//        std::cout << e.what() << std::endl;
//    }

    std::cout << "onnxruntime inference end" << std::endl;
    // Get pointer to output tensor float values
    float* pdata = ort_outputs[0].GetTensorMutableData<float>();


    /*for (int i = 0; i < 100; i++) {
        std::cout << pdata[i] << std::endl;
    }*/

    const Ort::TensorTypeAndShapeInfo& shapeInfo = ort_outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> sp = shapeInfo.GetShape();
    std::cout << "onnxruntime out length: " << sp << std::endl;	// 1 * t_out * 512


    std::vector<vector<float>> out_n_512;
    int pdata_index;
    for (int j = 0; j < sp[1]; j++) {
        vector<float> encoder_512;
        for (int k = 0; k < sp[2]; k++) {
            pdata_index = j * 512 + k;
            encoder_512.push_back(pdata[pdata_index]);
        }
        out_n_512.push_back(encoder_512);
    }

    vector<vector<vector<float>>> out_mel;


    for (int m = 0; m < out_n_512.size() - 4;m++) {
        std::vector<vector<float>> out_5_512(out_n_512.begin() + m, out_n_512.begin() + m +5);
        out_mel.push_back(out_5_512);
    }

    int mel_length = out_mel.size();
//    torch::Tensor mel_tensor = torch::from_blob(out_mel.data(), { mel_length,1, 5, 512 });
    cout << "tensor copy start " << endl;
    clock_t tensor_gen_start = clock();
    torch::Tensor mel_tensor = torch::empty({ mel_length, 1, 5, 512 });

    float* data_ptr = mel_tensor.data_ptr<float>();
    std::memcpy(data_ptr, out_mel.data(), out_mel.size() * sizeof(std::vector<std::vector<float>>));

    clock_t tensor_gen_end = clock();
    cout << "tensor copy end " << endl;
    cout << "tenso copy cost : " << tensor_gen_end - tensor_gen_start << endl;
    cout << mel_tensor.sizes() << endl;
    cout << mel_tensor.dim() << endl;
    torch::Tensor mel_tensor_cuda = mel_tensor.to(torch::kCUDA);

    cv::FileStorage fs2("../c_file/template/2023-04-21_22-06-37_24422621726989746_changed_head_fusion_vid.yml", cv::FileStorage::READ);
    cv::Mat face_det_boxes;
    fs2["face_det_boxes"] >> face_det_boxes;
    int frameFps = (int)fs2["fps"];
    int frameHeight = (int)fs2["frame_h"];
    int frameWidth = (int)fs2["frame_w"];

    // 构建face信息
    VideoCapture capture("../c_file/template/2023-04-21_22-06-37_24422621726989746_changed_head_fusion_vid.mp4");
    cout << "Cap open " << capture.isOpened() << endl;
    if (!capture.isOpened())
    {
        std::cout << "please check camera !!!" << std::endl;
        return false;
    }

    vector<Mat> frames_batch;
    vector<Mat> faces_batch;
    cv::Mat frame;
    cout << "mel_length " << mel_length << endl;
//    torch::Tensor frames_3_tensor = torch::zeros({ mel_length, 256, 256, 3 }, torch::kFloat32);
    torch::Tensor frames_3_tensor = torch::empty({ mel_length, 256, 256,3 }, torch::kByte);
    for (int n = 0; n < out_mel.size(); n++) {
        capture >> frame;
        frames_batch.push_back(frame);

        //拷贝构造函数2---只拷贝感兴趣的区域----由Rect对象定义
        //rect左上角（100，100），宽高均为200，（x,y,width,height)
        int y1 = face_det_boxes.at<int>(n, 0);
        int y2 = face_det_boxes.at<int>(n, 1);
        int x1 = face_det_boxes.at<int>(n, 2);
        int x2 = face_det_boxes.at<int>(n, 3);
        //cv::Rect rect(x1, y1, x2 - x1, y2 - y1);
        cv::Mat frame_face(frame, cv::Rect(x1, y1, x2 - x1, y2 - y1));
        cv::resize(frame_face, frame_face, cv::Size(256, 256));
        //frame_face.convertTo(frame_face, CV_32FC3, 1.0 / 255.0);

//        faces_batch.push_back(frame_face);
        std::memcpy(frames_3_tensor[n].data_ptr(), frame_face.data, frame_face.total() * frame_face.elemSize());
    }
    frames_3_tensor = frames_3_tensor.to(torch::kFloat32).div_(255).to(torch::kCUDA);
    cout << "frames_3_tensor done" << endl;
    //torch::Tensor frames_3_tensor_u8 = torch::from_blob(faces_batch.data(), ((int)out_mel.size(), 256, 256, 3), torch::dtype(torch::kFloat32));
    //auto ffff_c = frames_3_tensor_u8.clone();


    //torch::Tensor frames_3_tensor_u8_cuda = torch::from_blob(faces_batch.data(), ((int)out_mel.size(), 256, 256, 3), torch::dtype(torch::kFloat32).device(torch::kCUDA));
    //auto ffff_c_cuda = frames_3_tensor_u8_cuda.clone();


    //torch::Tensor frames_3_tensor_u8 = torch::from_blob(faces_batch.data(), ((int)out_mel.size(), 256, 256, 3), torch::dtype(torch::kFloat32).device(torch::kCUDA));
    //torch::Tensor frames_3_tensor = torch::from_blob(faces_batch.data(), ((int)out_mel.size(), 256, 256, 3 ),torch::device(torch::kCUDA)).to(torch::kFloat32);
//    torch::Tensor frames_3_tensor = torch::from_blob(faces_batch.data(), ((int)out_mel.size(), 256, 256, 3 )).to(torch::kFloat32);
    torch::Tensor frames_3_tensor_mask = frames_3_tensor.clone();
    //torch::Tensor frames_3_tensor_cuda = frames_3_tensor.to(torch::kCUDA);
    //torch::Tensor frames_3_tensor_mask_cuda = frames_3_tensor_mask.to(torch::kCUDA);

    frames_3_tensor_mask.index_put_({Slice(), Slice(128), Slice(), Slice() }, 0);
    torch::Tensor frames_6_tensor_cuda = torch::cat({ frames_3_tensor, frames_3_tensor_mask }, 3).permute({0, 3, 1, 2}).to(torch::kCUDA);

    std::string model_pb = "../c_file/checkpoints/wav2lip_c_cuda.pt";
    auto module = torch::jit::load(model_pb);
    module.to(at::kCUDA);
    std::cout << "loaded." << std::endl;


    // ls change
//    std::vector<torch::jit::IValue> inputs;
//    auto inputs = c10::impl::GenericList(c10::StringType::get());
//    inputs.push_back(mel_tensor_cuda);
//    inputs.push_back(frames_6_tensor_cuda);

//    auto outputs = module.forward({ inputs }).toTuple();
    auto outputs = module.forward({ mel_tensor_cuda, frames_6_tensor_cuda }).toTuple();
    torch::Tensor out1 = outputs->elements()[0].toTensor();
    torch::Tensor out2 = outputs->elements()[1].toTensor();

    std::cout << "pre infer." << std::endl;


    session_options.release();
    session_.release();
    capture.release();

    return true;
}

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

void audioCallback(int rows, int cols, int stride, float* feats) {
//    cout << "rows : " << rows << endl;
//    cout << "stride : " << stride << endl;
//    cout << "feats add: " << feats << endl;
//    cout << "***** extra audio feature *****" << endl;
    wav2lip(rows, cols, stride, feats);
}

int main()
{
//    bool we_result = infer_wenet();
//    bool yolov8_result = infer_yolov8();
//    bool libtorch_result = infer_libtorch();
    //bool ymlResult = test_yml();
    //test_cv_vid();

    bool suc = extraAudioFeature_File(&audioCallback, AUDIO_PATH);
    cout << "suc : " << suc << endl;
    std::cout << "done" << std::endl;
}
