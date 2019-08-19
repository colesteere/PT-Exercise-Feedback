# Physical Therapy (PT) Exercise Assistant

This is an assistant that provides just-in-time instructions for PT exercises. Here are demo videos:
https://www.youtube.com/playlist?list=PLMP5vmwBuCAik6TUq5KkvUzZ9OUWoq_BC

## Citations
The majority of this code comes from https://github.com/ildoonet/tf-pose-estimation. I wrote various scripts that allow specific exercise detection and feedback which uses the code from tf-pose-estimation. Go to this repository in order to install/understand how to run the basic code, or just navigate to the bottom of this README for basic instructions. 

# tf-pose-estimation

'Openpose', human pose estimation algorithm, have been implemented using Tensorflow. It also provides several variants that have some changes to the network structure for **real-time processing on the CPU or low-power embedded devices.**

## Demo


*********************************************************************************************************************************
If you need help understanding how to download and use the original tf-pose-estimation, go to https://github.com/ildoonet/tf-pose-estimation. The following will include instructions on how to get the scripts and servers I wrote to run.


### Instructions

## Webcam
1. In order to get the webcam version working, you either need to have a built in webcam in your laptop/computer, or have an external webcam plugged in. Once you have a webcam, navigate to the main tf-pose-estimation directory. For me, it was called tf-pose-estimation.

```
~/tf-pose-estimation$ 
```

2. After that, depending on whether you want to start the hip abduction helper or bicep curl helper, type either one of the following:

```
$ python run_bicep_curl.py --resize=256x256
```
or

```
$ python run_hip_abductor.py --resize=256x256
```

3. Quit the script with ctrl + C.

## Server (Use with Gabriel's mobile client)
1. Navigate to the main tf-pose-estimation directory. For me, it was called tf-pose-estimation.

```
~/tf-pose-estimation$ 
```

2. After that, you can decide whether or not you want to run the server with bicep curl or hip abductor feedback. Set whichever variable at the top of the server file to true that you want to run. Next, start the server. Remember that you need to start the client side after starting this server in order for the mobile application to actually work. The instructions to do that can be found in the READ ME in the repository called "updated-gabriel-proxy-server."

```
$ python pairserver.py
```


3. Quit the server with ctrl + C.


*********************************************************************************************************************************
The information found below can also be found at https://github.com/ildoonet/tf-pose-estimation.

## Install

### Dependencies

You need dependencies below.

- python3
- tensorflow 1.4.1+
- opencv3, protobuf, python3-tk
- slidingwindow
  - https://github.com/adamrehn/slidingwindow
  - I copied from the above git repo to modify few things.

### Pre-Install Jetson case

```bash
$ sudo apt-get install libllvm-7-ocaml-dev libllvm7 llvm-7 llvm-7-dev llvm-7-doc llvm-7-examples llvm-7-runtime
$ export LLVM_CONFIG=/usr/bin/llvm-config-7 
```

### Install

Clone the repo and install 3rd-party libraries.

```bash
$ git clone https://www.github.com/ildoonet/tf-pose-estimation
$ cd tf-pose-estimation
$ pip3 install -r requirements.txt
```

Build c++ library for post processing. See : https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess
```
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

### Package Install

Alternatively, you can install this repo as a shared package using pip.

```bash
$ git clone https://www.github.com/ildoonet/tf-pose-estimation
$ cd tf-pose-estimation
$ python setup.py install  # Or, `pip install -e .`
```

## Models & Performances

See [experiments.md](./etc/experiments.md)

### Download Tensorflow Graph File(pb file)

Before running demo, you should download graph files. You can deploy this graph on your mobile or other platforms.

- cmu (trained in 656x368)
- mobilenet_thin (trained in 432x368)
- mobilenet_v2_large (trained in 432x368)
- mobilenet_v2_small (trained in 432x368)

CMU's model graphs are too large for git, so I uploaded them on an external cloud. You should download them if you want to use cmu's original model. Download scripts are provided in the model folder.

```
$ cd models/graph/cmu
$ bash download.sh
```

## Demo

### Test Inference

You can test the inference feature with a single image.

```
$ python run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg
```

The image flag MUST be relative to the src folder with no "~", i.e:
```
--image ../../Desktop
```

Then you will see the screen as below with pafmap, heatmap, result and etc.

![inferent_result](./etcs/inference_result2.png)

### Realtime Webcam

```
$ python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0
```

Apply TensoRT 

```
$ python run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0 --tensorrt=True
```

Then you will see the realtime webcam screen with estimated poses as below. This [Realtime Result](./etcs/openpose_macbook13_mobilenet2.gif) was recored on macbook pro 13" with 3.1Ghz Dual-Core CPU.

## Python Usage

This pose estimator provides simple python classes that you can use in your applications.

See [run.py](run.py) or [run_webcam.py](run_webcam.py) as references.

```python
e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
humans = e.inference(image)
image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
```

If you installed it as a package,

```python
import tf_pose
coco_style = tf_pose.infer(image_path)
```

## ROS Support

See : [etcs/ros.md](./etcs/ros.md)

## Training

See : [etcs/training.md](./etcs/training.md)

## References

See : [etcs/reference.md](./etcs/reference.md)
