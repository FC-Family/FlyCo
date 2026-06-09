# FlyCo

## *Planner & Simulator*

✍️ Authors: [Chen Feng](https://chen-albert-feng.github.io/AlbertFeng.github.io/) and [Guiyong Zheng](https://gy920.github.io/)

This workspace provides the planning module and simulation platform used in **F**ly**C**o, including prediction-aware hierarchical planning, viewpoint-constrained trajectory optimization, and a physics-realistic simulator.

### 🛠️ Installation

**Prerequisite:**

* ROS Noetic (Ubuntu 20.04)
* PCL 1.10
* [Eigen 3.4.1](https://hkustconnect-my.sharepoint.com/:u:/g/personal/cfengag_connect_ust_hk/ES7krJtO3E1Oh4wY0-Wcr-gBDZ3dWz9bpbFNKp6Yhpn3Yg?e=mfiKrO)
* Farthest Point Sampling

```shell
  cd src/skeleton_decomp/fps/sampling_cpu/
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --config Release
```

* Ray Tracing

1. [embree-3](https://github.com/embree/embree/releases/tag/v3.12.2)
2. [libigl](https://github.com/libigl/libigl)

* Trajectory Optimization

```shell
  sudo apt update
  sudo apt install cpufrequtils
  sudo apt install libompl-dev
```

* AirSim Simulator

```
  git clone -b 4.25 git@github.com:EpicGames/UnrealEngine.git
  cd UnrealEngine
  ./Setup.sh
  ./GenerateProjectFiles.sh
  make

  git clone https://github.com/Microsoft/AirSim.git
  cd AirSim
  ./setup.sh
  ./build.sh
```

Modify your ```~/Documents/AirSim/settings.json``` as the same as [setting.json](./script/setting/settings.json).

**Compilation:**

* caktin tools

```shell
  source src/3rdparty/embree-3/embree.zsh (source src/3rdparty/embree-3/embree.bash)
  catkin config -DCMAKE_BUILD_TYPE=Release
  catkin build --cmake-args -Wno-dev
```

If you have installed ***Anaconda*** or ***Miniconda***, please use ``catkin build --cmake-args -Wno-dev -DPYTHON_EXECUTABLE=/usr/bin/python3``.

* caktin make

```shell
  source src/3rdparty/embree-3/embree.zsh (source src/3rdparty/embree-3/embree.bash)
  catkin_make --cmake-args -Wno-dev
```

If you have installed ***Anaconda*** or ***Miniconda***, please use ``catkin_make --cmake-args -Wno-dev -DPYTHON_EXECUTABLE=/usr/bin/python3``.

### ⚖️ License

This software is released under the PolyForm Noncommercial License 1.0.0. Third-party components retain their original licenses.