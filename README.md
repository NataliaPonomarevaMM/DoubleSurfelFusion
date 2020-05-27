# [DoubleSurfelFusion]

DoubleSurfelFusion is a dynamic reconstruction of human.

### Build Instructions

The code was originally developed with `CUDA 10.1`.
You need to build im2smpl project according to [the guidance](https://github.com/ZhengZerong/im2smpl/blob/master/README.md). This project is used for initialization of SMPL model. Also you need to build PCL or Cilantro library for visualization.

### Run Instructions

First, follow instructions for [SurfelWarp project](https://github.com/weigao95/surfelwarp/blob/master/README.md). For BodyFusionVicon check [website](http://www.liuyebin.com/doublefusion/doublefusion_software.htm). Then build project and run script `run.sh`. If everything goes well, the executable would produce the reconstructed result per frame in folder `results`. 
