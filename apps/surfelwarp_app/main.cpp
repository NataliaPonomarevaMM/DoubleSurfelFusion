//
// Created by wei on 5/22/18.
//

#include "common/common_utils.h"
#include "common/ConfigParser.h"
#include "core/SurfelWarpSerial.h"
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <cstdlib>
#include <chrono>

using mcs = std::chrono::microseconds;
using ms = std::chrono::milliseconds;
using clk = std::chrono::system_clock;

int main(int argc, char** argv) {
	using namespace surfelwarp;
	
	//Get the config path
	std::string config_path;
	if (argc <= 1) 
		config_path = "/home/wei/Documents/programs/surfelwarp/test_data/boxing_config.json";
	else
		config_path = std::string(argv[1]);

	//Parse it
	auto& config = ConfigParser::Instance();
	config.ParseConfig(config_path);

	//The context
	//auto context = initCudaContext();

	//Save offline
	bool offline_rendering = true;

    auto begin = clk::now();
	//The processing loop
	SurfelWarpSerial fusion;
    auto end = clk::now();
    auto time = std::chrono::duration_cast<ms>(end - begin);
    std::cout << ") time " << (double)time.count() << std::endl;

    begin = clk::now();
    //The processing loop
    fusion.ProcessFirstFrame();
    end = clk::now();
    time = std::chrono::duration_cast<ms>(end - begin);
    std::cout << ") time " << (double)time.count() << std::endl;

	double sr = 0;
	for (auto i = 0; i < config.num_frames(); i++){
		LOG(INFO) << "The " << i << "th Frame";
        auto begin = clk::now();
        fusion.ProcessNextFrameWithReinit(offline_rendering);
        auto end = clk::now();
        auto time = std::chrono::duration_cast<ms>(end - begin);
        sr += (double)time.count();
        std::cout << i << ") time " << (double)time.count() << std::endl;
	}
    sr /= config.num_frames();
    std::cout << "SR: " << sr << " mcs" << std::endl;
	
	//destroyCudaContext(context);
}
