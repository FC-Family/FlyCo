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

import numpy as np
import sensor_msgs.point_cloud2 as pc2

from .cluster_postprocess import IncrementalClusterConfig


class ClusterTargetCache:
    def __init__(self, incremental_config=None):
        self.incremental_config = incremental_config or IncrementalClusterConfig()
        self._accumulated_cluster_points = np.empty((0, 3), dtype=np.float32)
        self._cluster_target_pc = None
        self._latest_cluster_pc = None

    @property
    def cluster_target_pc(self):
        return self._cluster_target_pc

    @property
    def latest_cluster_pc(self):
        return self._latest_cluster_pc

    def has_target(self):
        return self._cluster_target_pc is not None or self._latest_cluster_pc is not None

    def reset(self):
        self._accumulated_cluster_points = np.empty((0, 3), dtype=np.float32)
        self._cluster_target_pc = None
        self._latest_cluster_pc = None

    def ingest(self, cluster_msg, cluster_postprocess):
        if cluster_msg is None or len(cluster_msg.data) == 0:
            return None
        self._latest_cluster_pc = cluster_msg

        if self.incremental_config.enabled:
            cluster_target_pc, accumulated_points = (
                cluster_postprocess.build_incremental_target(
                    cluster_msg=cluster_msg,
                    accumulated_points=self._accumulated_cluster_points,
                    config=self.incremental_config,
                )
            )
            self._accumulated_cluster_points = np.asarray(
                accumulated_points, dtype=np.float32
            )
            if cluster_target_pc is not None:
                self._cluster_target_pc = pc2.create_cloud_xyz32(
                    cluster_msg.header, cluster_target_pc
                )
                return self._cluster_target_pc

        self._cluster_target_pc = cluster_msg
        return self._cluster_target_pc
