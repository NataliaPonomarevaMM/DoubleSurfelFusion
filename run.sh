#!/bin/bash


cp ~/data/BodyFuViconDataset/sqz2/color/color_frame_100.png ~/data/testimage/test.jpg
cd im2smpl
python3.8 main.py --img_file ~/data/testimage/test.jpg --out_dir ~/data/
cd ..
./build/apps/surfelwarp_app/surfelwarp_app test_data/boxing_config.json
