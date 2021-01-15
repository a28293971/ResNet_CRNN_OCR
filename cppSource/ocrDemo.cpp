#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <chrono>


std::string decode(const torch::Tensor& singleTensor, const std::string& alphabets) {
	if (singleTensor.dim() != 1)
		throw std::exception("the input tensor must be a single dim");

	auto cpuTensor = singleTensor.cpu();
	auto accr = cpuTensor.accessor<int64, 1>();

	std::string resStr;
	// the index of '-' is 0
	for (int64 i = 0; i < singleTensor.size(0); ++i)
		if (accr[i] != 0 && !(i && accr[i - 1] == accr[i]))
			resStr += alphabets[accr[i] - 1];
	return resStr;
}

int main() {
	const std::string modelPath = "PATH/TO/YOUR/TORCH/SCIPT/MODEL/FILE",
		imgPath = "PATH/TO/YOUR/IMAGE";

	const auto device = torch::kCPU; //or torch::kCUDA
	const int imageH = 32, imageW = 160;
	const double normStd = 0.193, normMean = 0.588;
	const std::string alphabets = "1234567890-"; // remember '-'

	if (device == torch::kCUDA && !torch::cuda::is_available())
		throw std::exception("this device dose not support GPU");

	auto module = torch::jit::load(modelPath, device);
	module.eval();

	{ //limit the scope of NoGradGuard
		puts("pre run model");
		torch::NoGradGuard no_grad;
		module.forward({ torch::randn({ 1, 1, imageH, imageW }, c10::TensorOptions().device(device)) });
		puts("pre run done...");
	}

	auto st = std::chrono::high_resolution_clock::now();
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
	cv::resize(img, img, { imageW, imageH });
	img.convertTo(img, CV_32F, 1.0 / 255.0);

	auto input_tensor = torch::from_blob(img.data, { 1, imageH, imageW, 1 }, c10::TensorOptions().device(device));
	input_tensor = input_tensor.permute({ 0, 3, 1, 2 });
	input_tensor = torch::data::transforms::Normalize<>(normMean, normStd)(input_tensor);

	
	torch::NoGradGuard no_grad;
	auto output = module.forward({ input_tensor }).toTensor();
	auto maxRes = std::get<1>(torch::max(output, 2));

	std::string strRes = decode(maxRes.permute({ 1, 0 })[0], alphabets);
	std::cout << "total use time:" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - st).count() << "ms\n"
		<< "the predicted result is: " << strRes << std::endl;

	return 0;
}