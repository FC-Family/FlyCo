# ⭐⭐⭐******************************************************************⭐⭐⭐
# Authors      :    Chen Feng <cfengag at connect dot ust dot hk>, UAV Group, ECE, HKUST.
#                   Guiyong Zheng <shverses at gmail dot com>, SAI, SYSU.
# Homepage     :    https://chen-albert-feng.github.io/AlbertFeng.github.io/
#                   https://gy920.github.io/
# Date         :    Jun. 2026
# E-mail       :    cfengag at connect dot ust dot hk
#                   shverses at gmail dot com
# Description  :    This file is part of the FlyCo Perception public ROS runtime.
# Copyright    :    Copyright (c) 2026 Chen Feng and Guiyong Zheng.
# License      :    PolyForm Noncommercial License 1.0.0
#                   <https://polyformproject.org/licenses/noncommercial/1.0.0/>
# Project      :    FlyCo: Foundation Model-Empowered Drones for Autonomous 3D Structure Scanning in Open-World Environments
# Website      :    https://hkust-aerial-robotics.github.io/FC-Planner/
# ⭐⭐⭐******************************************************************⭐⭐⭐

import rospy

from mesh_predict.ros.node import MeshPredictNode
from mesh_predict.ros.params import load_node_params
from mesh_predict.runtime.resources import load_official_runtime_spec


def main(node_name="mesh_predict_node"):
    """Boot the official mesh_predict ROS node."""

    rospy.init_node(name=node_name)
    params = load_node_params()
    runtime_spec = load_official_runtime_spec(
        params.runtime_config_path, params.mesh_pred_ckpt_path
    )
    node = MeshPredictNode(params=params, runtime_spec=runtime_spec)
    node.run()
