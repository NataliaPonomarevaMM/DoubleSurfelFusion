//
// Created by wei on 5/22/18.
//

#include "common/ConfigParser.h"
#include "core/SurfelWarpSerial.h"
#include <iostream>

int main(int argc, char** argv) {
	using namespace surfelwarp;
	
	//Get the config path
	std::string config_path = std::string(argv[1]);

	//Parse it
	auto& config = ConfigParser::Instance();
	config.ParseConfig(config_path);

	//Save offline
	bool offline_rendering = false;

	//The processing loop
	SurfelWarpSerial fusion;

    //The processing loop
    fusion.ProcessFirstFrame();
	for (auto i = 0; i < config.num_frames(); i++){
		LOG(INFO) << "The " << i << "th Frame";
        fusion.ProcessNextFrameWithReinit(offline_rendering);
	}
}
