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

import cv2
import numpy as np
import rospy
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray as Int32MultiArrayMsg
from std_msgs.msg import MultiArrayDimension, String as StringMsg


class PromptUiCv:
    def __init__(self, node_name: str) -> None:
        rospy.init_node(node_name)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/img_topic", Image, self.image_callback)
        self.points_pub = rospy.Publisher(
            "/query_points", Int32MultiArrayMsg, queue_size=10
        )
        self.text_pub = rospy.Publisher("/query_text", StringMsg, queue_size=10)
        self.text_sub = rospy.Subscriber("/query_text", StringMsg, self.text_callback)

        self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.image_dirty = False
        self.points = []
        self.boxes = []
        self.drawing_box = False
        self.box_start = None
        self.box_end = None
        self.help_visible = False
        self.text_prompt = rospy.get_param("~initial_text_prompt", "")
        self.text_input_active = False
        self.text_input_buffer = self.text_prompt
        self.status_height = 124
        self.button_rects = {}
        self.active_button = ""
        self.button_flash_until = 0.0
        self.text_rect = None

        self.window_name = "FlyCo-UI"
        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL,
        )
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def image_callback(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.image = cv2.resize(image, (640, 480))
            self.image_dirty = True
        except Exception as exc:
            rospy.logerr("Failed to convert image: %s", exc)

    def publish_points(self) -> None:
        if not self.points:
            rospy.loginfo("No points to publish.")
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

    def publish_text(self, text: str) -> None:
        if not text:
            return
        msg = StringMsg()
        msg.data = text
        self.text_pub.publish(msg)
        rospy.loginfo("Published text: %s", text)
        self.text_prompt = text
        self.text_input_buffer = text

    def text_callback(self, msg: StringMsg) -> None:
        text = msg.data.strip()
        if not text:
            return
        self.text_prompt = text
        if not self.text_input_active:
            self.text_input_buffer = text

    @staticmethod
    def point_in_rect(x: int, y: int, rect) -> bool:
        if rect is None:
            return False
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def draw_button(self, status, rect, label: str, active: bool) -> None:
        fill = (225, 238, 255) if active else (245, 245, 245)
        border = (235, 145, 40) if active else (190, 190, 190)
        cv2.rectangle(status, (rect[0], rect[1]), (rect[2], rect[3]), fill, -1)
        cv2.rectangle(status, (rect[0], rect[1]), (rect[2], rect[3]), border, 2 if active else 1)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        text_x = rect[0] + max(0, (rect[2] - rect[0] - text_size[0]) // 2)
        text_y = rect[1] + max(text_size[1] + 2, (rect[3] - rect[1] + text_size[1]) // 2)
        cv2.putText(
            status,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )

    def handle_control_click(self, x: int, y: int) -> None:
        if self.point_in_rect(x, y, self.button_rects.get("send_points")):
            self.flash_button("send_points")
            self.publish_points()
            return
        if self.point_in_rect(x, y, self.button_rects.get("clear_points")):
            self.flash_button("clear_points")
            self.clear()
            return
        if self.point_in_rect(x, y, self.button_rects.get("send_text")):
            self.flash_button("send_text")
            self.text_input_active = False
            self.publish_text(self.text_input_buffer.strip())
            return
        if self.point_in_rect(x, y, self.text_rect):
            self.text_input_active = True
            self.text_input_buffer = self.text_prompt

    def flash_button(self, name: str) -> None:
        self.active_button = name
        self.button_flash_until = time.time() + 0.18

    def draw_overlay(self):
        canvas = self.image.copy()
        for x, y, flag in self.points:
            if flag == 1:
                color = (0, 0, 255)
            elif flag == 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            cv2.circle(canvas, (x, y), 5, color, -1)

        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if self.drawing_box and self.box_start and self.box_end:
            cv2.rectangle(canvas, self.box_start, self.box_end, (0, 255, 255), 1)

        status = np.full((self.status_height, canvas.shape[1], 3), (250, 250, 250), dtype=np.uint8)
        if self.help_visible:
            status_text = (
                "Left:+  Right:-  Middle drag:bbox  d:publish  "
                "t/click text:edit  Enter:publish  c:clear  h:hide  q:quit"
            )
        else:
            status_text = (
                "Left:+  Right:-  Middle:bbox  d:publish  "
                "t/click text:edit  Enter:publish  c:clear  h:help  q:quit"
            )
        prompt_label = "Text*:" if self.text_input_active else "Text:"
        prompt_text = self.text_input_buffer if self.text_input_active else self.text_prompt
        if len(prompt_text) > 78:
            prompt_text = "..." + prompt_text[-75:]
        button_y1 = 38
        button_y2 = 72
        self.button_rects = {
            "send_points": (8, button_y1, 132, button_y2),
            "clear_points": (142, button_y1, 266, button_y2),
            "send_text": (276, button_y1, 400, button_y2),
        }
        if self.active_button and time.time() > self.button_flash_until:
            self.active_button = ""
        for name, label in (
            ("send_points", "send points"),
            ("clear_points", "clear points"),
            ("send_text", "send text"),
        ):
            self.draw_button(status, self.button_rects[name], label, name == self.active_button)

        self.text_rect = (8, 84, canvas.shape[1] - 8, 114)
        cv2.rectangle(status, (self.text_rect[0], self.text_rect[1]), (self.text_rect[2], self.text_rect[3]), (255, 255, 255), -1)
        cv2.rectangle(
            status,
            (self.text_rect[0], self.text_rect[1]),
            (self.text_rect[2], self.text_rect[3]),
            (235, 145, 40) if self.text_input_active else (205, 205, 205),
            1,
        )
        cv2.putText(
            status,
            status_text,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            status,
            f"{prompt_label} {prompt_text}",
            (16, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        return np.vstack((canvas, status))

    def mouse_callback(self, event, x, y, _flags, _param):
        if x < 0 or x >= self.image.shape[1] or y < 0:
            return
        if y >= self.image.shape[0]:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.handle_control_click(x, y - self.image.shape[0])
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y, 1))
            rospy.loginfo("Added positive point: (%s, %s)", x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append((x, y, 0))
            rospy.loginfo("Added negative point: (%s, %s)", x, y)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.drawing_box = True
            self.box_start = (x, y)
            self.box_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing_box:
            self.box_end = (x, y)
        elif event == cv2.EVENT_MBUTTONUP and self.drawing_box:
            self.drawing_box = False
            self.box_end = (x, y)
            if self.box_start and self.box_end:
                x1 = min(self.box_start[0], self.box_end[0])
                y1 = min(self.box_start[1], self.box_end[1])
                x2 = max(self.box_start[0], self.box_end[0])
                y2 = max(self.box_start[1], self.box_end[1])
                self.points.append((x1, y1, 2))
                self.points.append((x2, y2, 3))
                self.boxes.append((x1, y1, x2, y2))
                rospy.loginfo("Added bbox: (%s, %s, %s, %s)", x1, y1, x2, y2)
            self.box_start = None
            self.box_end = None

    def clear(self) -> None:
        self.points.clear()
        self.boxes.clear()
        self.box_start = None
        self.box_end = None
        self.drawing_box = False
        rospy.loginfo("Cleared points and boxes.")

    def run(self) -> None:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            canvas = self.draw_overlay()
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if self.text_input_active:
                if key in (10, 13):
                    self.text_input_active = False
                    self.publish_text(self.text_input_buffer.strip())
                elif key in (8, 127):
                    self.text_input_buffer = self.text_input_buffer[:-1]
                elif key == 27:
                    self.text_input_active = False
                    self.text_input_buffer = self.text_prompt
                elif 32 <= key <= 126:
                    self.text_input_buffer += chr(key)
                rate.sleep()
                continue
            if key == ord("q"):
                break
            if key == ord("c"):
                self.clear()
            elif key == ord("d"):
                self.publish_points()
            elif key == ord("t"):
                self.text_input_active = True
                self.text_input_buffer = self.text_prompt
            elif key == ord("h"):
                self.help_visible = not self.help_visible
            rate.sleep()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        PromptUiCv("click_node").run()
    except rospy.ROSInterruptException:
        pass
