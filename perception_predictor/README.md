# FlyCo

## *Perception & Predictor*

✍️ Authors: [Chen Feng](https://chen-albert-feng.github.io/AlbertFeng.github.io/) and [Guiyong Zheng](https://gy920.github.io/)

This workspace provides the perception and prediction side of **F**ly**C**o, including prompt-driven 2D target tracking, 3D target clustering, and mesh prediction for downstream planning and visualization.

### 🛠️ Installation

**Prerequisite:**

* ROS Noetic (Ubuntu 20.04)
* CUDA-capable GPU
* Conda / Miniconda
* PyTorch matching your CUDA driver

Create and activate the Python environment:

```shell
  conda create -n flyco python=3.8 -y
  conda activate flyco
```

Install PyTorch following the official command for your CUDA version. For example, for CUDA 11.8:

```shell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Install ROS and Python dependencies:

```shell
  sudo apt update
  sudo apt install python3-catkin-tools python3-rosdep ros-noetic-cv-bridge ros-noetic-pcl-ros ros-noetic-image-transport ros-noetic-tf2-ros

  cd ${YOUR_PERCEPTION_PREDICTOR_PATH}
  pip install -r src/sam_tracking/requirements.txt
  pip install -r src/mesh_predict/requirements.txt
```

Install SAM2:

```shell
  cd ${YOUR_PERCEPTION_PREDICTOR_PATH}/src/sam_tracking/thirdparty/sam2-realtime
  pip install -e .
```

Download the SAM2 tracking checkpoints:

```shell
  cd ${YOUR_PERCEPTION_PREDICTOR_PATH}/src/sam_tracking
  bash ckpt/download_ckpt.sh
```

Prepare mesh prediction checkpoints under:

```shell
  ${YOUR_PERCEPTION_PREDICTOR_PATH}/src/mesh_predict/resources/checkpoints/
```

The mesh prediction checkpoints are available from [Google Drive](https://drive.google.com/file/d/1dYMss4kwzxXIMgyFlBQQXxU79y8GA5EQ/view?usp=sharing).

Expected local files are documented in [src/mesh_predict/resources/checkpoints/README.md](src/mesh_predict/resources/checkpoints/README.md). The runtime can also resolve custom checkpoint paths from:

* `MESH_PREDICT_MESH_CKPT`
* `MESH_PREDICT_NKSR_CKPT`

Build the ROS workspace:

```shell
  cd ${YOUR_PERCEPTION_PREDICTOR_PATH}
  source /opt/ros/noetic/setup.bash
  catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python
  source devel/setup.bash
```

If you prefer `catkin tools`:

```shell
  cd ${YOUR_PERCEPTION_PREDICTOR_PATH}
  source /opt/ros/noetic/setup.bash
  catkin config -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python
  catkin build --cmake-args -Wno-dev
  source devel/setup.bash
```

### ⚖️ License

This software is released under the PolyForm Noncommercial License 1.0.0. Third-party components retain their original licenses.
