import numpy as np
import cv2
import os
from PIL import Image
from collections import deque
from scipy.ndimage import binary_dilation

_PALETTE = (
    [0, 0, 0]
    + [255, 0, 0]
    + [0, 255, 0]
    + [0, 0, 255]
    + [255, 255, 0]
    + [255, 0, 255]
    + [0, 255, 255]
) + [0] * (256 * 3 - 21)


def mask2bbox(mask):
    if len(np.where(mask > 0)[0]) == 0:
        return np.array([[0, 0], [0, 0]]).astype(np.int64)

    x_ = np.sum(mask, axis=0)
    y_ = np.sum(mask, axis=1)

    x0 = np.min(np.nonzero(x_)[0])
    x1 = np.max(np.nonzero(x_)[0])
    y0 = np.min(np.nonzero(y_)[0])
    y1 = np.max(np.nonzero(y_)[0])

    return np.array([[x0, y0], [x1, y1]]).astype(np.int64)


def draw_outline(mask, frame):
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

    return frame


def draw_points(points, modes, frame):
    neg_points = points[np.argwhere(modes == 0)[:, 0]]
    pos_points = points[np.argwhere(modes == 1)[:, 0]]

    for i in range(len(neg_points)):
        point = neg_points[i]
        cv2.circle(frame, (point[0], point[1]), 8, (255, 80, 80), -1)

    for i in range(len(pos_points)):
        point = pos_points[i]
        # print(point)
        cv2.circle(frame, (point[0], point[1]), 8, (0, 153, 255), -1)

    return frame


def save_prediction(pred_mask, output_dir, file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode="P")
    save_mask.putpalette(_PALETTE)
    save_mask.save(os.path.join(output_dir, file_name))


def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode="P")
    save_mask.putpalette(_PALETTE)
    save_mask = save_mask.convert(mode="RGB")
    return np.array(save_mask)


def draw_mask(img, mask, alpha=0.5, id_countour=False):

    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _PALETTE[id * 3 : id * 3 + 3]
            else:
                color = [0, 0, 0]
            foreground = img * (1 - alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = mask == id

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = mask != 0
        countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours, :] = 0

    return img_mask.astype(img.dtype)


def create_dir(dir_path):
    # if os.path.isdir(dir_path):
    #     os.system(f"rm -r {dir_path}")

    # os.makedirs(dir_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def detections_update(mask, score=0.9):
    """
    Generates a detections array from a given binary mask with a default confidence score.

    Parameters:
    - mask: numpy array (h, w), where mask is expected to be a binary image with objects marked by non-zero pixel values.
    - score: float, the confidence score for the detected object.

    Returns:
    - detections: numpy array of shape (1, 5), formatted as [[x1, y1, x2, y2, score]]
    """
    # Find all non-zero points in the mask
    points = np.where(mask != 0)
    if points[0].size == 0:
        # No objects found in the mask
        return np.empty((0, 5))

    # Calculate bounding box coordinates for the object
    x_min = np.min(points[1])
    y_min = np.min(points[0])
    x_max = np.max(points[1])
    y_max = np.max(points[0])

    # Create a detections array with the calculated bounding box and the specified score
    detections = np.array([[x_min, y_min, x_max, y_max, score]])

    return detections


def get_mid_of_detections(detections):
    """
    Calculate the midpoint coordinates from detections and convert them to integer values.

    Parameters:
    - detections: numpy array of shape (n, 5), formatted as [[x1, y1, x2, y2, score], ...]

    Returns:
    - coords: numpy array of shape (1, 2), formatted as [[x_mid, y_mid]], where x_mid and y_mid are integers
    """
    if detections.size == 0:
        return np.empty((0, 2))  # 返回空数组，如果没有检测

    # Calculate midpoints for all detections
    x_mid = (detections[:, 0] + detections[:, 2]) / 2
    y_mid = (detections[:, 1] + detections[:, 3]) / 2

    # Choose the first midpoint for simplicity or you can choose based on conditions
    mid_coords = np.array([[x_mid[0], y_mid[0]]]).astype(int)  # Convert to integer

    return mid_coords


def get_mid_of_bbox(bboxes):
    """
    Calculate the midpoint coordinates from bounding boxes and convert them to integer values.

    Parameters:
    - bboxes: numpy array of shape (n, 4), formatted as [[x1, y1, x2, y2], ...]

    Returns:
    - coords: numpy array of shape (n, 2), formatted as [[x_mid, y_mid]], where x_mid and y_mid are integers
    """
    if bboxes.size == 0:
        return np.empty((0, 2))  # Return an empty array if there are no bounding boxes

    # Calculate midpoints for all bounding boxes
    x_mid = (bboxes[:, 0] + bboxes[:, 2]) / 2
    y_mid = (bboxes[:, 1] + bboxes[:, 3]) / 2

    # Convert to integer and format as [[x_mid, y_mid]] for each box
    mid_coords = np.stack([x_mid, y_mid], axis=1).astype(int)

    return mid_coords


def get_multiple_midpoints_of_mask(mask):
    """
    Calculate nine key points from an object in a binary mask: center point, midpoints of edges, and quarter points towards each corner.

    Parameters:
    - mask: numpy array (h, w), binary mask where object is marked by non-zero values.

    Returns:
    - coords: list of lists formatted as [[x_mid, y_mid], [x1_mid, y1_mid], ..., [x8_mid, y8_mid], ...]
    """
    # Find contours of the mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []  # Return an empty list if no contours are found

    # Assuming the largest contour corresponds to the main object
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    x2, y2 = x + w, y + h

    # Calculate center points and midpoints
    x_center = (x + x2) / 2
    y_center = (y + y2) / 2

    x1_mid, y1_mid = (x_center + x) / 2, y_center
    x2_mid, y2_mid = (x_center + x2) / 2, y_center
    x3_mid, y3_mid = x_center, (y_center + y) / 2
    x4_mid, y4_mid = x_center, (y_center + y2) / 2
    x5_mid, y5_mid = (x_center + x) / 2, (y_center + y) / 2
    x6_mid, y6_mid = (x_center + x2) / 2, (y_center + y) / 2
    x7_mid, y7_mid = (x_center + x) / 2, (y_center + y2) / 2
    x8_mid, y8_mid = (x_center + x2) / 2, (y_center + y2) / 2

    # Collect all points
    coords = [
        [int(x_center), int(y_center)],
        [int(x1_mid), int(y1_mid)],
        [int(x2_mid), int(y2_mid)],
        [int(x3_mid), int(y3_mid)],
        [int(x4_mid), int(y4_mid)],
        [int(x5_mid), int(y5_mid)],
        [int(x6_mid), int(y6_mid)],
        [int(x7_mid), int(y7_mid)],
        [int(x8_mid), int(y8_mid)],
    ]

    return coords


def get_uniform_points_inside_contour(mask, num_points=10):
    """
    Calculate uniform points inside the largest contour in a binary mask.

    Parameters:
    - mask: numpy array (h, w), binary mask where object is marked by non-zero values.
    - num_points: int, approximate number of points to calculate inside the largest contour.

    Returns:
    - points: list of lists formatted as [[x1, y1], [x2, y2], ..., [xn, yn]]
    """
    # Find contours of the mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []  # Return an empty list if no contours are found

    # Assuming the largest contour corresponds to the main object
    contour = max(contours, key=cv2.contourArea)

    # Create a blank canvas and draw the filled contour
    stencil = np.zeros_like(mask)
    cv2.drawContours(stencil, [contour], -1, (255), thickness=cv2.FILLED)

    # Calculate the bounding rectangle to limit the sample area
    x, y, w, h = cv2.boundingRect(contour)
    step_x = int(np.sqrt((w * h) / num_points))
    step_y = step_x
    if step_x <= 0:
        step_x = 1
        step_y = 1

    # Generate points within the bounding rectangle
    points = []
    for i in range(x, x + w, step_x):
        for j in range(y, y + h, step_y):
            if cv2.pointPolygonTest(contour, (i, j), False) >= 0:
                points.append([i, j])

    return points


def adjust_coords_to_mask(coords, mask, buffer=80):
    """
    Adjust the coordinates to ensure they are within the valid areas of the last_mask and a specified buffer area
    around each coordinate does not cross the mask boundary. This check is done using a circular buffer.

    Parameters:
    - coords: numpy array of shape (n, 2), formatted as [[x_mid, y_mid], ...]
    - last_mask: numpy array (h, w), mask where 1 indicates valid areas.
    - buffer: int, the radius of the buffer area around each coordinate to check for boundary.

    Returns:
    - adjusted_coords: numpy array of shape (n, 2), adjusted coordinates
    """
    h, w = mask.shape
    adjusted_coords = np.zeros_like(coords)

    for i, (x, y) in enumerate(coords):
        # Check if the coordinate is within bounds
        if x >= 0 and x < w and y >= 0 and y < h:
            # Create a circular mask to check the buffer area
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - x) ** 2 + (Y - y) ** 2)
            circular_mask = dist_from_center <= buffer

            # Check if the circular buffer area is entirely within the mask
            if np.all(mask[circular_mask]):
                adjusted_coords[i] = [x, y]  # Coordinates are valid within buffer
            else:
                # Adjust coordinates by finding the nearest valid point within the circular mask
                valid_points = np.where(mask & circular_mask)
                if valid_points[0].size > 0:
                    distances = (valid_points[1] - x) ** 2 + (valid_points[0] - y) ** 2
                    min_idx = np.argmin(distances)
                    adjusted_coords[i] = [
                        valid_points[1][min_idx],
                        valid_points[0][min_idx],
                    ]
                else:
                    adjusted_coords[i] = [
                        x,
                        y,
                    ]  # Use original if no valid points found within buffer
        else:
            adjusted_coords[i] = [x, y]  # Use original coordinates if out of bounds

    return adjusted_coords


def adjust_coords_to_mask_simple(coords, mask):
    """
    确保坐标完全位于mask的有效区域内。如果坐标位于无效区域,则移动到最近的有效点。

    参数:
    - coords: numpy数组,形状为(n, 2)，格式为[[x_mid, y_mid], ...]
    - mask: numpy数组(h, w),掩码,1表示有效区域。

    返回:
    - adjusted_coords: numpy数组,形状为(n, 2)，调整后的坐标
    """

    h, w = mask.shape
    adjusted_coords = np.zeros_like(coords)

    for i, (x, y) in enumerate(coords):
        # 检查坐标是否已经在有效区域内
        if 0 <= x < w and 0 <= y < h and mask[y, x]:
            adjusted_coords[i] = [x, y]
        else:
            # 如果坐标在无效区域，找到最近的有效点
            y_indices, x_indices = np.nonzero(mask)  # 获取所有有效点的坐标
            distances = (x_indices - x) ** 2 + (y_indices - y) ** 2
            min_idx = np.argmin(distances)
            adjusted_coords[i] = [x_indices[min_idx], y_indices[min_idx]]

    return adjusted_coords


def get_coords_from_mask(mask, last_mask, edge_threshold=25, max_attempts=5):
    """
    确保坐标完全位于mask和last_mask的有效区域内。如果坐标位于无效区域,则重新采样直到找到有效点。

    参数:
    - mask: numpy数组(h, w), 当前掩码, 1表示有效区域。
    - last_mask: numpy数组(h, w), 上一个掩码, 1表示有效区域。
    - edge_threshold: 边缘检查的阈值。
    - max_attempts: 最大尝试次数。

    返回:
    - adjusted_coords: numpy数组, 形状为(n, 2)，调整后的坐标
    """
    h, w = mask.shape
    attempts = 0
    points_num = 10
    while attempts < max_attempts:
        coords = get_uniform_points_inside_contour(mask, points_num)
        valid_coords = []

        for x, y in coords:
            if 0 <= x < w and 0 <= y < h and mask[y, x] == 1 and last_mask[y, x] == 1:
                angles = np.linspace(0, 2 * np.pi, 9)[:-1]
                edge_points = [
                    (
                        int(x + edge_threshold * np.cos(angle)),
                        int(y + edge_threshold * np.sin(angle)),
                    )
                    for angle in angles
                ]
                if all(
                    0 <= px < w
                    and 0 <= py < h
                    and mask[py, px] == 1
                    and last_mask[py, px] == 1
                    for px, py in edge_points
                ):
                    valid_coords.append([x, y])

        if valid_coords:
            return np.array(valid_coords)

        attempts += 1
        points_num += 10

    # 如果尝试了max_attempts次仍没有找到有效点，返回空数组
    return np.array([[]])


def draw_and_save_adjusted_coords(image, adjusted_coords, frame_idx, save_path):
    """
    Draws adjusted coordinates on the image and saves it.

    Parameters:
    - image: numpy array (h, w, 3), the original image in BGR color space.
    - adjusted_coords: numpy array of shape (n, 2), adjusted coordinates to be drawn.
    - save_path: str, path where the modified image will be saved.
    """
    # Create a copy of the image to draw on
    output_image = image.copy()

    # Loop through each coordinate and draw a circle
    for coord in adjusted_coords:
        x, y = coord
        cv2.circle(output_image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    # Save the image to the specified path
    save_path = f"{save_path}/frame_{frame_idx:05d}.jpg"
    cv2.imwrite(save_path, output_image)
    # print(f"Image saved to {save_path}")


def convert_to_rectangle_coordinates(bbox):
    """
    Converts a bounding box from a 2D numpy array into two points format expected by cv2.rectangle and other functions.

    Parameters:
    - bbox: 2D numpy array or list, expected to contain at least one row with elements [x0, y0, x1, y1, score]

    Returns:
    - List of two tuples: [(x0, y0), (x1, y1)]

    Raises:
    - ValueError: If the bounding box array does not conform to expected shape or size.
    """
    if isinstance(bbox, np.ndarray):
        if bbox.ndim != 2 or bbox.shape[1] < 4:
            raise ValueError(
                "Bounding box array must have at least one row with four columns for coordinates."
            )
        # Extract the coordinates and convert to integer
        x0, y0, x1, y1 = map(int, bbox[0, :4])
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x0, y0, x1, y1 = map(int, bbox[:4])
    else:
        raise ValueError(
            "Bounding box must be a list, tuple, or numpy array with at least four elements."
        )

    return [(x0, y0), (x1, y1)]


def save_bbox_frame(frame, bbox, frame_idx, output_folder):
    """
    Process the frame by converting color space, drawing a bounding box from bbox, and saving to disk.

    Parameters:
    - frame: numpy array, the original frame in RGB color space.
    - bbox: numpy array, bbox data in the format [[x1, y1, x2, y2, score]].
    - frame_idx: int, index of the current frame for naming the saved file.
    - output_folder: str, path to the folder where the image will be saved.
    """
    if bbox.shape[0] == 0:
        return

    # Convert RGB frame to BGR for display and saving
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Draw each bounding box from bbox on the frame
    for det in bbox:
        x1, y1, x2, y2, score = map(int, det)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the processed frame to disk
    save_path = f"{output_folder}/frame_{frame_idx:05d}.jpg"
    cv2.imwrite(save_path, frame_bgr)
    # print(f"Frame saved to {save_path}")


def create_bbox_from_mask(mask):
    """
    Create a bounding box from a given binary mask.
    Arguments:
        mask: numpy array (h, w), where the mask is expected to be a binary image with
              objects marked by non-zero pixel values.
    Returns:
        A list of lists representing the bounding box: [[x0, y0], [x1, y1]]
    """
    # Find all non-zero points in the mask
    points = np.where(mask != 0)
    if points[0].size == 0:
        # No objects found in the mask
        return []

    # Calculate bounding box coordinates for the object
    x_min = np.min(points[1])
    y_min = np.min(points[0])
    x_max = np.max(points[1])
    y_max = np.max(points[0])

    # Create a bounding box list of lists
    bbox = [[x_min, y_min], [x_max, y_max]]

    return bbox


def create_bbox_from_mask_np(mask):
    """
    Create a bounding box from a given binary mask.
    Arguments:
        mask: numpy array (h, w), where the mask is expected to be a binary image with
              objects marked by non-zero pixel values.
    Returns:
        A list of lists representing the bounding box: [[x0, y0], [x1, y1]]
    """
    # Find all non-zero points in the mask
    points = np.where(mask != 0)
    if points[0].size == 0:
        # No objects found in the mask
        return []

    # Calculate bounding box coordinates for the object
    x_min = np.min(points[1])
    y_min = np.min(points[0])
    x_max = np.max(points[1])
    y_max = np.max(points[0])

    # Create a bounding box list of lists
    bbox = np.asarray([[x_min, y_min], [x_max, y_max]])

    return bbox


def cal_iou_from_masks(last_mask, pred_mask):
    """
    Calculate the Intersection over Union (IoU) between two masks by determining the bounding boxes of the masks.

    Parameters:
    - last_mask: numpy array, a binary mask where the objects are marked.
    - pred_mask: numpy array, another binary mask where the objects are marked.

    Returns:
    - iou: float, the intersection over union of the two mask areas.
    """
    # Calculate bounding boxes from each mask
    bbox_last_mask = create_bbox_from_mask(last_mask)
    bbox_pred_mask = create_bbox_from_mask(pred_mask)

    # Check if either bounding box is empty
    if not bbox_last_mask or not bbox_pred_mask:
        return 0.0

    # Convert bounding boxes to numpy arrays for calculation
    bbox_a = np.array(
        [
            bbox_last_mask[0][0],
            bbox_last_mask[0][1],
            bbox_last_mask[1][0],
            bbox_last_mask[1][1],
        ]
    )
    bbox_b = np.array(
        [
            bbox_pred_mask[0][0],
            bbox_pred_mask[0][1],
            bbox_pred_mask[1][0],
            bbox_pred_mask[1][1],
        ]
    )

    # Calculate areas of each bounding box
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    # Calculate intersection coordinates
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    # Calculate the area of intersection
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate IoU
    iou = (
        inter_area / float(area_a + area_b - inter_area)
        if (area_a + area_b - inter_area) > 0
        else 0
    )

    return iou


def cal_iou(bbox, pred_mask):
    """
    Calculate the Intersection over Union (IoU) between a bounding box and a mask.

    Parameters:
    - bbox: array-like, expected to have format [x0, y0, x1, y1] or [[x0, y0, x1, y1]]
    - pred_mask: numpy array, the prediction mask from which to calculate the bounding box

    Returns:
    - iou: float, the intersection over union of the two areas
    """
    # Check if bbox is empty
    if bbox.size == 0 or len(bbox) == 0:
        return 0.0
    # Normalize bbox input to [x0, y0, x1, y1]
    if isinstance(bbox[0], list) or isinstance(
        bbox[0], np.ndarray
    ):  # Handle [[x0, y0, x1, y1]]
        bbox = bbox[0]

    # Ensure bbox is correctly formatted
    if len(bbox) < 4:
        raise ValueError("Bounding box must contain at least four elements.")

    # Create bounding box from prediction mask
    bbox_from_mask = create_bbox_from_mask(pred_mask)
    if not bbox_from_mask:
        return 0.0  # If no bounding box could be created from the mask, return 0 IoU

    # Convert bounding boxes to numpy arrays
    bbox_a = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
    bbox_b = np.array(
        [
            bbox_from_mask[0][0],
            bbox_from_mask[0][1],
            bbox_from_mask[1][0],
            bbox_from_mask[1][1],
        ]
    )

    # Calculate areas
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    # Find intersection coordinates
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])

    # Calculate intersection area
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate IoU
    iou = inter_area / (area_a + area_b - inter_area)
    return iou


def fill_contours_in_pred_mask(pred_mask, kernel_size=5):
    """
    在预测掩膜的最大外层轮廓内填充，将该轮廓内的非零部分全部置为255。

    参数:
    - pred_mask: 预测掩膜，应为8位单通道灰度图像的NumPy数组。

    返回:
    - filled_mask: 填充后的掩膜，为NumPy数组。
    """
    # 检查掩膜数据类型
    if pred_mask.dtype != np.uint8:
        raise ValueError("Image must be 8-bit grayscale.")

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(pred_mask, kernel, iterations=1)
    # 查找最外层轮廓
    contours, _ = cv2.findContours(
        dilated_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 如果没有找到轮廓，则直接返回原掩膜
    if not contours:
        return pred_mask

    # 计算每个轮廓的面积并找到最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 创建一个与原掩膜相同大小的数组用于填充最大轮廓
    filled_mask = np.zeros_like(dilated_mask)
    cv2.drawContours(filled_mask, [max_contour], -1, (255,), thickness=cv2.FILLED)

    return filled_mask


class MaskQueue:
    def __init__(self, resample_size=10, max_size=20):
        self.queue = deque()
        self.resample_size = resample_size
        self.max_size = max_size

    def add(self, frame, mask):
        """添加新 mask 并在所有帧中保持均匀分布，保留第一帧。"""
        if not self.queue:  # 初始化时添加第一帧
            self.queue.append((frame, mask))
            return

        if len(self.queue) < self.max_size:
            self.queue.append((frame, mask))
        else:
            self.queue = deque(self._resample(frame, mask))

    def _resample(self, new_frame, new_mask):
        """重新采样 mask 队列，以在所有帧中均匀分布并保持队列长度。"""
        # 确保第一帧保持不变
        resampled = [self.queue[0]]  # 保留第一帧

        frame_ids = range(len(self.queue) - self.resample_size, len(self.queue) - 1)

        step = (len(self.queue) - self.resample_size) // (self.max_size - 2)
        for i in range(1, self.resample_size - 1 - self.resample_size):
            frame_idx = i * step
            if frame_idx < len(self.queue):
                frame_ids.add(frame_idx)
        sorted(frame_ids)

        """
        step = len(self.queue) // (self.resample_size - 2)

        for i in range(1, self.resample_size - 1):
            frame_idx = i * step
            if frame_idx < len(self.queue):
                frame_ids.add(frame_idx)

        if (len(frame_ids) + 2) < self.resample_size:
            remaining_ids = [idx for idx in range(1, len(self.queue)) if idx not in frame_ids]

            frame_ids += remaining_ids[len(remaining_ids) - (self.resample_size - 2 - len(frame_ids)):]
            sorted(frame_ids)
        """

        for idx in frame_ids:
            resampled.append(self.queue[idx])

        resampled.append((new_frame, new_mask))

        return resampled

    def get(self):
        """获取当前 mask 队列，仅返回 mask 数据。"""
        return [frame for frame, _ in self.queue], [mask for _, mask in self.queue]
