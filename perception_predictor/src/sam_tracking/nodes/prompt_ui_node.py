#!/usr/bin/env python3

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

import sys

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray as Int32MultiArrayMsg
from std_msgs.msg import MultiArrayDimension, String as StringMsg


class PromptUi(QWidget):
    def __init__(self, node_name):
        super().__init__()
        rospy.init_node(node_name)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/img_topic", Image, self.image_callback)
        self.points_pub = rospy.Publisher(
            "/query_points", Int32MultiArrayMsg, queue_size=10
        )
        self.text_pub = rospy.Publisher("/query_text", StringMsg, queue_size=10)
        self.text_sub = rospy.Subscriber("/query_text", StringMsg, self.text_callback)

        self.image_queue = []
        self.points = []
        self.drawing_bbox = False
        self.bbox_start = None
        self.base_image = np.zeros((480, 640, 3), np.uint8)
        self.pixmap = QPixmap()
        self.initial_text_prompt = rospy.get_param("~initial_text_prompt", "")
        self.pending_text_prompt = ""

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_image)
        self.timer.start(30)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("FlyCo-UI")
        self.setGeometry(100, 100, 760, 620)

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.mousePressEvent = self.image_clicked
        self.image_label.mouseReleaseEvent = self.image_released
        self.set_image(self.base_image)

        font = QFont("Times New Roman", 13)
        self.text_edit = QLineEdit(self)
        self.text_edit.setPlaceholderText('please input text prompt, like "a tower"')
        self.text_edit.setFont(font)
        if self.initial_text_prompt:
            self.text_edit.setText(self.initial_text_prompt)

        self.send_points_button = QPushButton("send points", self)
        self.send_points_button.setFont(font)
        self.send_points_button.clicked.connect(self.send_points)

        self.clear_button = QPushButton("clear points", self)
        self.clear_button.setFont(font)
        self.clear_button.clicked.connect(self.clear_points)

        self.send_text_button = QPushButton("send text", self)
        self.send_text_button.setFont(font)
        self.send_text_button.clicked.connect(self.send_text)

        button_row = QHBoxLayout()
        button_row.addWidget(self.send_points_button)
        button_row.addWidget(self.clear_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addLayout(button_row)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.send_text_button)
        self.setLayout(layout)

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            rospy.logerr("Failed to convert image: %s", exc)
            return
        if image is None:
            return
        self.image_queue.append(cv2.resize(image, (640, 480)))

    def set_image(self, image_bgr):
        height, width, channel = image_bgr.shape
        bytes_per_line = channel * width
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        q_image = QImage(
            image_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        self.pixmap = QPixmap.fromImage(q_image.copy())
        self.image_label.setPixmap(self.pixmap)

    def refresh_image(self):
        if self.image_queue:
            self.base_image = self.image_queue[-1].copy()
            self.image_queue.clear()
        if self.pending_text_prompt:
            self.text_edit.setText(self.pending_text_prompt)
            self.pending_text_prompt = ""
        self.draw_points()

    def publish_points(self):
        if not self.points:
            rospy.loginfo("No points to send")
            return
        msg = Int32MultiArrayMsg()
        flat = [coord for point in self.points for coord in point]

        dim_points = MultiArrayDimension()
        dim_points.label = "points"
        dim_points.size = len(self.points)
        dim_points.stride = len(flat)

        dim_coords = MultiArrayDimension()
        dim_coords.label = "coords"
        dim_coords.size = 3
        dim_coords.stride = 3

        msg.layout.dim = [dim_points, dim_coords]
        msg.layout.data_offset = 0
        msg.data = flat
        self.points_pub.publish(msg)
        rospy.loginfo("Published points: %s", self.points)

    def publish_text(self, text):
        if not text:
            return
        msg = StringMsg()
        msg.data = text
        self.text_pub.publish(msg)
        rospy.loginfo("Published text: %s", text)

    def text_callback(self, msg):
        text = msg.data.strip()
        if text:
            self.pending_text_prompt = text

    def clear_points(self):
        self.points.clear()
        self.bbox_start = None
        self.drawing_bbox = False
        self.draw_points()

    def send_points(self):
        self.publish_points()

    def send_text(self):
        self.publish_text(self.text_edit.text().strip())

    def draw_points(self):
        self.set_image(self.base_image)
        painter = QPainter(self.pixmap)

        for x, y, label in self.points:
            if label == 1:
                painter.setPen(QPen(Qt.red, 5))
            elif label == 0:
                painter.setPen(QPen(Qt.blue, 5))
            else:
                painter.setPen(QPen(Qt.green, 5))
            painter.drawPoint(x, y)

        for idx in range(0, len(self.points), 2):
            if idx + 1 >= len(self.points):
                continue
            x0, y0, label0 = self.points[idx]
            x1, y1, label1 = self.points[idx + 1]
            if label0 == 2 and label1 == 3:
                painter.setPen(QPen(Qt.green, 2))
                painter.drawRect(x0, y0, x1 - x0, y1 - y0)

        painter.end()
        self.image_label.setPixmap(self.pixmap)

    def label_pos_to_image_point(self, pos, clamp=False):
        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return None

        pixmap_w = pixmap.width()
        pixmap_h = pixmap.height()
        label_w = self.image_label.width()
        label_h = self.image_label.height()
        offset_x = max(0, (label_w - pixmap_w) // 2)
        offset_y = max(0, (label_h - pixmap_h) // 2)
        x = pos.x() - offset_x
        y = pos.y() - offset_y

        if clamp:
            x = min(max(x, 0), pixmap_w - 1)
            y = min(max(y, 0), pixmap_h - 1)
            return int(x), int(y)
        if x < 0 or y < 0 or x >= pixmap_w or y >= pixmap_h:
            return None
        return int(x), int(y)

    def image_clicked(self, event):
        point = self.label_pos_to_image_point(event.pos())
        if point is None:
            rospy.loginfo("Ignored click outside image area")
            return
        x, y = point
        if event.button() == Qt.MiddleButton:
            self.drawing_bbox = True
            self.bbox_start = (x, y)
            return
        if event.button() == Qt.LeftButton:
            self.points.append((x, y, 1))
            rospy.loginfo("Added positive point: (%s, %s)", x, y)
        elif event.button() == Qt.RightButton:
            self.points.append((x, y, 0))
            rospy.loginfo("Added negative point: (%s, %s)", x, y)
        self.draw_points()

    def image_released(self, event):
        if event.button() != Qt.MiddleButton or not self.drawing_bbox:
            return
        self.drawing_bbox = False
        if self.bbox_start is None:
            return
        x0, y0 = self.bbox_start
        point = self.label_pos_to_image_point(event.pos(), clamp=True)
        if point is None:
            return
        x1, y1 = point
        x_min, y_min = min(x0, x1), min(y0, y1)
        x_max, y_max = max(x0, x1), max(y0, y1)
        self.points.append((x_min, y_min, 2))
        self.points.append((x_max, y_max, 3))
        rospy.loginfo(
            "Added bbox: start=(%s, %s, 2), end=(%s, %s, 3)",
            x_min,
            y_min,
            x_max,
            y_max,
        )
        self.bbox_start = None
        self.draw_points()


def main():
    app = QApplication(sys.argv)
    ui = PromptUi("prompt_ui")
    ui.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
