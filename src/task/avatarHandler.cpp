//
// Created by Orangels on 2023/9/6.
//
#include "task/avatarHandler.h"
#include <ATen/core/function_schema.h>

#ifdef _WIN32
#else

#include <locale>
#include <codecvt>

#endif

using namespace std;
using namespace cv;
using namespace torch::indexing;

avatarHandler::avatarHandler() {
    auto conf = config_A->getConfig();
    //init cap, writer

    //init libtorch wav2lip
    string waveTolipModulePath = conf["WAV2LIP"]["MODULE_PATH"].as<string>();
    mWav2lip_model = torch::jit::load(waveTolipModulePath);
    mWav2lip_model.to(at::kCUDA);
    std::cout << "loaded wave2lip module." << std::endl;

    std::string parsingModulePath = conf["PARSING"]["MODULE_PATH"].as<string>();
    mParsing_model = torch::jit::load(parsingModulePath);
    mParsing_model.to(at::kCUDA);
    std::cout << "loaded parsing module." << std::endl;

    //init onnxruntime
    // 创建InferSession, 查询支持硬件设备
    // GPU Mode, 0 - gpu device id

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
    auto conf = config_A->getConfig();

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

    int max_data_size = 14;
    int total_batch_size = sp[1];
    int cycles_nums = total_batch_size / max_data_size;
    //余数
    int remainder = total_batch_size % max_data_size;
    int cycles_nums_total =  remainder == 0 ? cycles_nums : cycles_nums + 1;
    int end_tag = max_data_size;
    float* pdata_tmp;

    cout << "max_data_size : " << max_data_size << endl;
    cout << "total_batch_size : " << total_batch_size << endl;
    cout << "cycles_nums : " << cycles_nums << endl;
    cout << "remainder " << remainder << endl;
    cout << "cycles_nums_total : " << cycles_nums_total << endl;

    VideoWriter w1;
    string out_put_file = conf["TEST"]["OUTPUT_FILE"].as<string>() + "w2l_result_total.mp4";
    w1.open(out_put_file, VideoWriter::fourcc('D', 'I', 'V', 'X'), 25, Size(1080, 1920), true);   //创建保存视频文件的视频流  img.size()为保存的视频图像的尺寸
    //cv::Mat frame_all_cv;

    torch::NoGradGuard no_grad;

    cv::FileStorage fs2(conf["PROGRAME"]["FACE_BBOX_PATH"][0].as<string>(), cv::FileStorage::READ);
    cv::Mat face_det_boxes;
    fs2["face_det_boxes"] >> face_det_boxes;
    int frameFps = (int)fs2["fps"];
    int frameHeight = (int)fs2["frame_h"];
    int frameWidth = (int)fs2["frame_w"];


    for (size_t p_data_i = 0; p_data_i < cycles_nums_total; p_data_i++)
    {
        if (p_data_i == 0)
        {
            pdata_tmp = pdata;
        }
        else if (p_data_i < cycles_nums) {
            pdata_tmp = pdata + (512 * (p_data_i * max_data_size - 4));
        }
        else {
            pdata_tmp = pdata + (512 * cycles_nums * max_data_size - 4);
            end_tag = remainder + 4;
        }

        std::vector<vector<float>> out_n_512;
        int pdata_index;
        for (int j = 0; j < end_tag; j++) {
            vector<float> encoder_512;
            for (int k = 0; k < sp[2]; k++) {
                pdata_index = j * 512 + k;
                encoder_512.push_back(pdata_tmp[pdata_index]);
                //cout << pdata[pdata_index] << " ";
            }
            //cout << endl;
            out_n_512.push_back(encoder_512);
        }

        /*std::vector<std::vector<float>> out_n_512(sp[1], std::vector<float>(sp[2]));
        std::memcpy(out_n_512.data(), pdata, sp[1] * sp[2] * sizeof(float));*/

        vector<vector<vector<float>>> out_mel;
        int out_n_512_size = end_tag;
        torch::Tensor mel_tensor = torch::zeros({ (long)out_n_512.size() - 4, 5, 512 }, torch::kFloat32).contiguous();
        //linux
        for (int m = 0; m < out_n_512_size - 4; m++) {
            std::vector<vector<float>> out_5_512(out_n_512.begin() + m, out_n_512.begin() + m + 5);
            out_mel.push_back(out_5_512);
            for (int i = 0; i < out_5_512.size(); ++i) {
                const float* src_ptr = out_5_512[i].data();
                float* dst_ptr = mel_tensor[m][i].data_ptr<float>();
                std::memcpy(dst_ptr, src_ptr, sizeof(float) * 512);
            }
        }
        mel_tensor.unsqueeze_(1);
        int mel_length = out_mel.size();

        //windows
        /*for (int m = 0; m < out_n_512_size - 4; m++) {
            std::vector<vector<float>> out_5_512(out_n_512.begin() + m, out_n_512.begin() + m + 5);
            out_mel.push_back(out_5_512);
        }
        int mel_length = out_mel.size();
        torch::Tensor mel_tensor = torch::from_blob(out_mel.data(), { mel_length,1, 5, 512 });*/

        cout << "tensor copy start " << endl;
        clock_t tensor_gen_start = clock();
        clock_t tensor_gen_end = clock();
        cout << "tensor copy end " << endl;
        cout << "tenso copy cost : " << tensor_gen_end - tensor_gen_start << endl;
        cout << mel_tensor.sizes() << endl;
        cout << mel_tensor.dim() << endl;
        cout << out_mel[0][0][0] << endl;
        cout << "*************" << endl;
        cout << mel_tensor[0][0][0][0].item<float>() << endl;
        torch::Tensor mel_tensor_cuda = mel_tensor.to(torch::kCUDA);

        std::cout << "tensor size:  " << mel_tensor.sizes() << std::endl;	// mel_length * 1 * 5 * 512

        // 构建face信息
        VideoCapture capture(conf["PROGRAME"]["FACE_VID_PATH"][0].as<string>());
        //VideoCapture capture("C:\\ls-dev\\ls_resourse\\c_file\\template\\test.mp4");
        //int start_frame_num = p_data_i * max_data_size;
        int start_frame_num = p_data_i * mel_length;
        capture.set(CAP_PROP_POS_FRAMES, start_frame_num);
        cout << "video frams count : " << capture.get(cv::CAP_PROP_FRAME_COUNT) << endl;
        cout << "video fps : " << capture.get(cv::CAP_PROP_FPS) << endl;
        cout << "Video start frame : " << start_frame_num << endl;
        cout << "Video end frame : " << start_frame_num + mel_length << endl;

        if (!capture.isOpened())
        {
            std::cout << "please check camera !!!" << std::endl;
            return false;
        }

        vector<Mat> frames_batch;
        vector<Mat> faces_batch;
        /*cv::Mat frame;*/
        torch::Tensor frames_3_tensor = torch::empty({ mel_length, 256, 256,3 }, torch::kByte);

        for (int n = 0; n < out_mel.size(); n++) {
            cv::Mat frame;
            capture >> frame;
            frames_batch.push_back(frame);

            //拷贝构造函数2---只拷贝感兴趣的区域----由Rect对象定义
            //rect左上角（100，100），宽高均为200，（x,y,width,height)
            int y1 = face_det_boxes.at<int>(start_frame_num + n, 0);
            int y2 = face_det_boxes.at<int>(start_frame_num + n, 1);
            int x1 = face_det_boxes.at<int>(start_frame_num + n, 2);
            int x2 = face_det_boxes.at<int>(start_frame_num + n, 3);
            //cv::Rect rect(x1, y1, x2 - x1, y2 - y1);
            cv::Mat frame_face(frame, cv::Rect(x1, y1, x2 - x1, y2 - y1));
            cv::resize(frame_face, frame_face, cv::Size(256, 256));
            //frame_face.convertTo(frame_face, CV_32FC3, 1.0 / 255.0);
            //faces_batch.push_back(frame_face);
            std::memcpy(frames_3_tensor[n].data_ptr(), frame_face.data, frame_face.total() * frame_face.elemSize());
            //frames_3_tensor[n] = torch::from_blob(frame_face.data, { frame_face.rows, frame_face.cols, 3 }, torch::kByte);
        }
        capture.release();
        //frames_3_tensor = frames_3_tensor.to(torch::kFloat32).div_(255).to(torch::kCUDA);
        frames_3_tensor = frames_3_tensor.to(torch::kFloat32).permute({ 0, 3, 1, 2 }).div_(255).to(torch::kCUDA);
        //frames_3_tensor = frames_3_tensor.to(torch::kFloat32).div_(255).to(torch::kCUDA);


        torch::Tensor frames_3_tensor_mask = frames_3_tensor.clone();

        torch::Tensor frames_6_tensor_cuda;
        try {
            frames_3_tensor_mask.index_put_({ Slice(), Slice(128), Slice(), Slice() }, 0);
            frames_6_tensor_cuda = torch::cat({ frames_3_tensor, frames_3_tensor_mask }, 1).to(torch::kCUDA);
        }
        catch (const c10::Error& e) {
            std::cout << "catch_error: " << e.msg() << std::endl;
        };

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(mel_tensor_cuda);
        inputs.push_back(frames_6_tensor_cuda);
        cout << "frames_6_tensor_cuda size " << frames_6_tensor_cuda.sizes() << endl;
        cout << "mel_tensor_cuda size " << mel_tensor_cuda.sizes() << endl;

        //auto outputs_ori = module.forward({ inputs });
        auto outputs_ori = mWav2lip_model.forward({ mel_tensor_cuda, frames_6_tensor_cuda });
        auto outputs = outputs_ori.toTuple();
        torch::Tensor wav2lip_out128 = outputs->elements()[0].toTensor();
        torch::Tensor wav2lip_out256 = outputs->elements()[1].toTensor();

        std::cout << wav2lip_out128.sizes() << endl;
        std::cout << wav2lip_out256.sizes() << endl;

        torch::Tensor wav2lip_out512 = torch::nn::functional::interpolate(wav2lip_out256,
                                                                          torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({ 512, 512 })).mode(torch::kBilinear));

        std::cout << "wav2lip_out512 infer." << wav2lip_out512.sizes() << wav2lip_out512.is_cuda() << std::endl;
        torch::Tensor wav2lip_out512_norm = wav2lip_out512.clone();
        wav2lip_out512_norm[0][0] = wav2lip_out512_norm[0][0].sub_(0.485).div_(0.229);
        wav2lip_out512_norm[0][1] = wav2lip_out512_norm[0][1].sub_(0.456).div_(0.224);
        wav2lip_out512_norm[0][2] = wav2lip_out512_norm[0][2].sub_(0.406).div_(0.225);

        torch::Tensor frames_3_tensor_re = torch::nn::functional::interpolate(frames_3_tensor, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({ 512, 512 })).mode(torch::kBilinear));
        torch::Tensor frames_3_tensor_re_norm = frames_3_tensor_re.clone();
        frames_3_tensor_re_norm[0][0] = frames_3_tensor_re_norm[0][0].sub_(0.485).div_(0.229);
        frames_3_tensor_re_norm[0][1] = frames_3_tensor_re_norm[0][1].sub_(0.456).div_(0.224);
        frames_3_tensor_re_norm[0][2] = frames_3_tensor_re_norm[0][2].sub_(0.406).div_(0.225);

        auto pred_parsing_out_all = mParsing_model.forward({ wav2lip_out512_norm }).toTuple();
        torch::Tensor out12 = pred_parsing_out_all->elements()[0].toTensor();
        torch::Tensor out22 = pred_parsing_out_all->elements()[1].toTensor();
        torch::Tensor out32 = pred_parsing_out_all->elements()[1].toTensor();
        std::cout << out12.sizes() << std::endl;
        std::cout << out22.sizes() << std::endl;
        std::cout << out32.sizes() << std::endl;
        torch::Tensor pred_parsing_out = out12.argmax(1);

        auto ori_parsing_out_all = mParsing_model.forward({ frames_3_tensor_re_norm }).toTuple();
        out12 = ori_parsing_out_all->elements()[0].toTensor();
        out22 = ori_parsing_out_all->elements()[1].toTensor();
        out32 = ori_parsing_out_all->elements()[1].toTensor();
        std::cout << out12.sizes() << std::endl;
        std::cout << out22.sizes() << std::endl;
        std::cout << out32.sizes() << std::endl;
        torch::Tensor ori_parsing_out = out12.argmax(1);

        torch::Tensor pred_label = torch::where((pred_parsing_out > 0) & (pred_parsing_out < 14) & (pred_parsing_out != 7) & (pred_parsing_out != 8) & (pred_parsing_out != 9), 1.0, 0.0).unsqueeze(1);
        torch::Tensor ori_label = torch::where((ori_parsing_out > 0) & (ori_parsing_out < 14) & (pred_parsing_out != 7) & (pred_parsing_out != 8) & (pred_parsing_out != 9), 1.0, 0.0).unsqueeze(1);

        std::cout << pred_label.sizes() << pred_label.dtype() << std::endl;
        std::cout << ori_label.sizes() << ori_label.dtype() << std::endl;

        // 创建ZeroPad2d层
        torch::nn::ZeroPad2d zero_pad(torch::nn::ZeroPad2dOptions(40));

        // 对输入张量进行ZeroPad2d操作
        torch::Tensor pred_label_pad = 1 - zero_pad->forward(pred_label);
        torch::Tensor ori_label_pad = 1 - zero_pad->forward(ori_label);
        std::cout << pred_label_pad.sizes() << std::endl;
        std::cout << ori_label_pad.sizes() << std::endl;


        torch::nn::MaxPool2d max_pool(torch::nn::MaxPool2dOptions(3).stride(1).padding(1));

        int iter_count = 4;
        for (int i = 0; i < iter_count; i++) {
            pred_label_pad = max_pool->forward(pred_label_pad);
        }

        torch::Tensor pred_label_pad_1 = 1 - pred_label_pad.index({ Slice(), Slice(), Slice(40, 512 + 40), Slice(40, 512 + 40) });

        torch::Tensor sigma = torch::empty(1).uniform_(0.1, 2.0);
        int kernel_value = 15;
        int kernel_size[2] = { kernel_value, kernel_value };

        float ksize_half = (kernel_value - 1) * 0.5;

        torch::Tensor x = torch::linspace(-ksize_half, ksize_half, kernel_value);
        torch::Tensor pdf = torch::exp(-0.5 * (x / sigma).pow(2)).to(at::kCUDA);
        torch::Tensor kernel1d = pdf / pdf.sum();
        torch::Tensor kernel2d = torch::mm(kernel1d.index({ Slice(), None }), kernel1d.index({ None, Slice() }));
        //std::cout << kernel2d << std::endl;
        kernel2d = kernel2d.expand({ 1,1,kernel_value, kernel_value });
        //std::cout << kernel2d.sizes() << std::endl;

        pred_label_pad_1 = torch::nn::functional::pad(pred_label_pad_1, torch::nn::functional::PadFuncOptions({ 7, 7, 7, 7 }).mode(torch::kReflect));
        torch::Tensor pred_label_gaus = torch::conv2d(pred_label_pad_1, kernel2d);

        torch::Tensor total_imgs = frames_3_tensor_re * (1 - pred_label_gaus) + wav2lip_out512 * pred_label_gaus;

        //std::cout << total_imgs.sizes() << std::endl;

        //VideoWriter w1;
        //w1.open("./result/temp1_result_62.mp4", VideoWriter::fourcc('D', 'I', 'V', 'X'), 25, Size(1080, 1920), true);   //创建保存视频文件的视频流  img.size()为保存的视频图像的尺寸
        torch::Tensor past_frames = total_imgs.permute({ 0, 2, 3, 1 }).mul_(255).clamp(0, 255).to(torch::kU8).to(torch::kCPU);
        cv::Mat frame_all_cv;

        for (int n = 0; n < mel_length; n++) {
            int y1 = face_det_boxes.at<int>(start_frame_num + n, 0);
            int y2 = face_det_boxes.at<int>(start_frame_num + n, 1);
            int x1 = face_det_boxes.at<int>(start_frame_num + n, 2);
            int x2 = face_det_boxes.at<int>(start_frame_num + n, 3);

            cv::Mat img_face_cv(512, 512, CV_8UC3);

            std::memcpy((void*)img_face_cv.data, past_frames[n].data_ptr(), sizeof(torch::kU8) * past_frames[n].numel());

            //cv::Mat o_Mat(cv::Size(width, height), CV_32F, i_tensor.data_ptr());


            cv::resize(img_face_cv, img_face_cv, cv::Size(x2 - x1, y2 - y1));

            frame_all_cv = frames_batch[n];
            img_face_cv.copyTo(frame_all_cv(cv::Rect(x1, y1, x2 - x1, y2 - y1)));

            /*string img_name = "./result/img_parsing/" + std::to_string(start_frame_num + n) + ".jpg";
            imwrite(img_name, frame_all_cv);*/

            w1 << frame_all_cv;
        }

        std::cout << "pre infer." << std::endl;
    }

    w1.release();
    return true;
}
