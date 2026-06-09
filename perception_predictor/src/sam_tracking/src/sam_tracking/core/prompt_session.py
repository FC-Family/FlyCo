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


class PromptSession:
    def __init__(self):
        self._query_points = []
        self._query_text = ""
        self._prompt_revision = 0
        self._consumed_prompt_revision = 0

    @staticmethod
    def _points_signature(query_points):
        return tuple(
            (
                int(getattr(point, "x", 0)),
                int(getattr(point, "y", 0)),
                int(getattr(point, "z", 1)),
            )
            for point in query_points
        )

    @staticmethod
    def _normalize_query_text(query_text):
        return " ".join(str(query_text or "").split())

    def set_query_points(self, query_points):
        query_points = list(query_points or [])
        if self._points_signature(query_points) != self._points_signature(
            self._query_points
        ):
            self._prompt_revision += 1
        self._query_points = query_points
        if not self._query_points:
            self._consumed_prompt_revision = self._prompt_revision

    def set_query_text(self, query_text):
        query_text = self._normalize_query_text(query_text)
        if query_text != self._query_text:
            self._prompt_revision += 1
        self._query_text = query_text

    @property
    def query_text(self):
        return self._query_text

    def has_prompt(self):
        return len(self._query_points) > 0 or bool(self._query_text.strip())

    @property
    def has_pending_provider_prompt_update(self):
        return self.has_prompt() and (
            self._prompt_revision > self._consumed_prompt_revision
        )

    def mark_provider_prompt_consumed(self):
        self._consumed_prompt_revision = self._prompt_revision

    def reset(self):
        self._query_points = []
        self._query_text = ""
        self._prompt_revision += 1
        self._consumed_prompt_revision = self._prompt_revision

    def as_provider_prompt(self):
        if not self.has_prompt():
            return None

        if self._query_points:
            coords = np.vstack([(int(p.x), int(p.y)) for p in self._query_points])
            labels = np.array(
                [int(getattr(p, "z", 1)) for p in self._query_points], dtype=np.int32
            )
        else:
            coords = np.zeros((0, 2), dtype=np.int32)
            labels = np.zeros((0,), dtype=np.int32)
        return {
            "coords": coords,
            "labels": labels,
            "text": self._query_text,
        }
