import pickle

import cv2
import matplotlib.pyplot as plt
import math
import skimage
from skimage.segmentation import slic
import numpy as np
from skimage.color import label2rgb

import time
import logging

from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm
import wandb
import csv
from collections import defaultdict

import argparse

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch, torchvision

logging.debug('torch version:', torch.__version__, torch.cuda.is_available())
logging.debug('torchvision version:', torchvision.__version__)

# Check MMPose installation
import mmpose

logging.debug('mmpose version:', mmpose.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version

logging.debug('cuda version:', get_compiling_cuda_version())
logging.debug('compiler information:', get_compiler_version())

from mmpose.apis import inference_topdown  # , process_mmdet_results
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmengine.registry import init_default_scope
import mmcv

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logging.info(f'Using: {device}')

def init_pose_estimation_model(config_path, checkpoint_path, device='cuda'):
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=False)))
    return init_pose_estimator(
        config_path,
        checkpoint_path,
        device=device,
        cfg_options=cfg_options
    )

def log_images(images, plot_title=None, subtitles=None):
    """
    Logs images with dynamic subplots (max 3 images per row) and individual subtitles to WandB.

    Parameters:
        images (list): A list of images (numpy arrays) to log.
        plot_title (str): The main title for the entire plot. If None, no main title is shown.
        subtitles (list): A list of subtitles corresponding to each image. If None, no subtitles are shown.
    """
    n_images = len(images)

    cols = min(n_images, 3)
    rows = math.ceil(n_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    if plot_title:
        fig.suptitle(plot_title, fontsize=16, y=1.02)

    if rows == 1:
        axes = [axes]
    axes = np.array(axes).flatten()

    for i, image in enumerate(images):
        if image is not None:
            axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[i].axis('off')

            if subtitles and i < len(subtitles):
                axes[i].set_title(subtitles[i], fontsize=12)
        else:
            axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    wandb.log({"images": wandb.Image(fig)})

    plt.close(fig)


durations = {}
def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logging.debug(f"{func.__name__} took {duration} seconds")
        if func.__name__ in durations:
            durations[func.__name__].append(duration)
        else:
            durations[func.__name__] = [duration]
        return result
    return wrapper

keypoints_history = defaultdict(lambda: defaultdict(list))

def check_overlap(bbox1, bbox2):
    """
    Checks whether two bounding boxes overlap.

    Parameters:
        bbox1 (tuple): (x1, y1, w1, h1) of the first bounding box.
        bbox2 (tuple): (x2, y2, w2, h2) of the second bounding box.

    Returns:
        bool: True if the bounding boxes overlap, False otherwise.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    return (x1 < x2 + w2 and
            x1 + w1 > x2 and
            y1 < y2 + h2 and
            y1 + h1 > y2)


def merge_boxes(bbox1, bbox2):
    """
    Merges two overlapping bounding boxes into a single bounding box.

    Parameters:
        bbox1 (tuple): (x1, y1, w1, h1) of the first bounding box.
        bbox2 (tuple): (x2, y2, w2, h2) of the second bounding box.

    Returns:
        tuple: The merged bounding box (x, y, w, h).
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate the new bounding box's top-left corner
    x = min(x1, x2)
    y = min(y1, y2)

    # Calculate the new bounding box's bottom-right corner
    x_right = max(x1 + w1, x2 + w2)
    y_bottom = max(y1 + h1, y2 + h2)

    # Calculate new width and height
    w = x_right - x
    h = y_bottom - y

    return (x, y, w, h)


def merge_overlapping_boxes(bounding_boxes):
    """
    Merges all overlapping bounding boxes in the list.

    Parameters:
        bounding_boxes (list): A list of bounding boxes, each defined as (x, y, w, h).

    Returns:
        list: A list of bounding boxes where all overlaps have been merged.
    """
    merged_boxes = []

    while bounding_boxes:
        current_box = bounding_boxes.pop(0)

        i = 0
        while i < len(bounding_boxes):
            # Check if the current box overlaps with any other box
            if check_overlap(current_box, bounding_boxes[i]):
                current_box = merge_boxes(current_box, bounding_boxes[i])
                bounding_boxes.pop(i)
            else:
                i += 1

        merged_boxes.append(current_box)

    return merged_boxes


def is_contained(smaller, bigger):
    """
    Checks if the smaller bounding box is fully contained within the bigger bounding box.

    Parameters:
        smaller (tuple): (x1, y1, w1, h1) of the smaller bounding box.
        bigger (tuple): (x2, y2, w2, h2) of the bigger bounding box.

    Returns:
        bool: True if the smaller bounding box is fully contained within the bigger bounding box.
    """
    x1, y1, w1, h1 = smaller
    x2, y2, w2, h2 = bigger

    # Check if all corners of the smaller box are inside the bigger box
    return (x1 >= x2 and
            y1 >= y2 and
            x1 + w1 <= x2 + w2 and
            y1 + h1 <= y2 + h2)


def remove_contained_boxes(bounding_boxes):
    """
    Removes larger bounding boxes if they fully contain smaller bounding boxes.

    Parameters:
        bounding_boxes (list): A list of bounding boxes, each defined as (x, y, w, h).

    Returns:
        list: A list of bounding boxes where larger boxes that contain smaller boxes are removed.
    """
    filtered_boxes = bounding_boxes.copy()

    for i in range(len(bounding_boxes)):
        for j in range(len(bounding_boxes)):
            if i != j and bounding_boxes[i] in filtered_boxes and bounding_boxes[j] in filtered_boxes:
                if is_contained(bounding_boxes[i], bounding_boxes[j]):
                    filtered_boxes.remove(bounding_boxes[j])

    return filtered_boxes


def crop_image_to_bounding_boxes(image, bounding_boxes, debug=False):
    """
    Crops an image based on a list of bounding boxes and returns a list of cropped images.

    Parameters:
        image (numpy.ndarray): The input image.
        bounding_boxes (list): A list of bounding boxes, each defined as (x, y, w, h).

    Returns:
        list: A list of cropped images, each corresponding to a bounding box.
    """
    cropped_images = []
    coordinates = []

    for bbox in bounding_boxes:
        x, y, w, h = bbox
        cropped_image = image[y:y + h, x:x + w]
        cropped_images.append(cropped_image)
        coordinates.append((x, y))
    if debug:
        log_images(cropped_images, plot_title="Cropped Images")

    return cropped_images, coordinates

@time_logger
def process_frames(frame_start, frame_end, scale_percent=100, debug=False):
    """
    Process two frames, calculate their difference, and optionally plot the frames.

    Parameters:
    - frame_start: The starting frame (numpy array).
    - frame_end: The ending frame (numpy array).
    - debug: If True, plots the frames and their difference.

    Returns:
    - frame_diff: The absolute difference between the two frames.
    """
    # Scale down the frames to 10% of their original size
    width_start = int(frame_start.shape[1] * scale_percent / 100)
    height_start = int(frame_start.shape[0] * scale_percent / 100)
    dim_start = (width_start, height_start)

    width_end = int(frame_end.shape[1] * scale_percent / 100)
    height_end = int(frame_end.shape[0] * scale_percent / 100)
    dim_end = (width_end, height_end)

    # Resize frames
    frame_start = cv2.resize(frame_start, dim_start, interpolation=cv2.INTER_AREA)
    frame_end = cv2.resize(frame_end, dim_end, interpolation=cv2.INTER_AREA)

    # Calculate the weighted sum for the starting frame
    frame_start = cv2.addWeighted(frame_start, 0.9, frame_start, 0, 0)

    # Calculate the absolute difference between the two frames
    frame_diff = cv2.absdiff(frame_start, frame_end)

    # Convert to grayscale and threshold the difference
    frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    frame_diff[frame_diff < 50] = 0
    frame_diff[frame_diff > 50] = 255

    # Convert back to BGR format
    frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR)

    if debug:
        log_images([frame_start, frame_end, frame_diff],
                   subtitles=["Frame Start", "Frame End", "Frame Difference"])

    return frame_diff, frame_end

@time_logger
def saliency_bounding_boxes(image, debug=False):
    """
    Applies static saliency detection using the Spectral Residual method, finds all contours,
    and filters bounding boxes by size. If multiple bounding boxes overlap, only the largest
    is kept, and bounding boxes are scaled by a fixed margin.

    Parameters:
        image (numpy.ndarray): The input image array in BGR format.
        debug (bool): If True, plots the original image with all bounding boxes visible.

    Returns:
        bounding_boxes (list): A list of (x, y, width, height) tuples for the scaled bounding boxes.
    """
    # Create a saliency object using the default spectral residual method
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()

    # Compute the saliency map
    #start_time = time.time()
    success, saliency_map = saliency.computeSaliency(image)
    saliency_map = (saliency_map * 255).astype("uint8")
    #logging.warning(f"Saliency detection took {time.time() - start_time} seconds.")

    # Threshold the saliency map to create a binary mask
    _, thresh_map = cv2.threshold(saliency_map, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours on the thresholded saliency map
    #start_time = time.time()
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #logging.warning(f"Contour detection took {time.time() - start_time} seconds.")

    bounding_boxes = []

    #start_time = time.time()
    if contours:
        # Use a list instead of a set to hold bounding boxes
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 50 and h >= 50:
                # Instead of using a set, directly append valid bounding boxes
                bounding_boxes.append((x, y, w, h))

        # Scaling and filtering overlapping bounding boxes
        scaled_bboxes = [
            (
                max(0, x - 125),
                max(0, y - 125),
                min(image.shape[1] - (x - 125), w + 250),
                min(image.shape[0] - (y - 125), h + 250)
            )
            for (x, y, w, h) in bounding_boxes
        ]

        # Merging overlapping bounding boxes in a more efficient way
        filtered_bounding_boxes = []
        used_indices = set()

        for i, (x, y, w, h) in enumerate(scaled_bboxes):
            if i in used_indices:
                continue

            largest_bbox = (x, y, w, h)
            largest_area = w * h

            for j, (x2, y2, w2, h2) in enumerate(scaled_bboxes):
                if i != j and (x < x2 + w2 and x + w > x2 and y < y2 + h2 and y + h > y2):
                    if w2 * h2 > largest_area:
                        largest_area = w2 * h2
                        largest_bbox = (x2, y2, w2, h2)
                    used_indices.add(j)

            filtered_bounding_boxes.append(largest_bbox)

        merged_bboxes = merge_overlapping_boxes(filtered_bounding_boxes)
        final_bboxes = remove_contained_boxes(merged_bboxes)

        if debug:
            log_images([image], plot_title="Saliency Detection Results")

        #logging.warning(f"Saliency bounding box detection took {time.time() - start_time} seconds.")
        return final_bboxes

    else:
        logging.debug("No contours found.")
        return []


def visualize_coco_keypoints(image, all_keypoints, point_size=3, line_size=2):
    """
    Visualizes pose estimation using a specified color-coding scheme.

    Parameters:
    - image: Input image as a NumPy array.
    - all_keypoints: A list of arrays, each containing shape (17, 2) for [x, y] for each keypoint.
    """

    # COCO keypoints mapping
    KEYPOINTS = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    }

    # Define colors
    COLORS = {
        'face': (255, 150, 0),  # Blue for face keypoints
        'left': (0, 165, 255),  # Orange for left side
        'right': (0, 255, 0),  # Green for right side
        'shoulder_hip': (255, 150, 0)  # Blue for shoulder-hip connections
    }

    # Define connections for the body parts
    CONNECTIONS = [
        # Face connections (blue)
        (KEYPOINTS['nose'], KEYPOINTS['left_eye']),
        (KEYPOINTS['nose'], KEYPOINTS['right_eye']),
        (KEYPOINTS['left_eye'], KEYPOINTS['left_ear']),
        (KEYPOINTS['right_eye'], KEYPOINTS['right_ear']),
        (KEYPOINTS['left_shoulder'], KEYPOINTS['left_ear']),
        (KEYPOINTS['right_shoulder'], KEYPOINTS['right_ear']),

        # Left side connections (orange)
        (KEYPOINTS['left_shoulder'], KEYPOINTS['left_elbow']),
        (KEYPOINTS['left_elbow'], KEYPOINTS['left_wrist']),
        (KEYPOINTS['left_hip'], KEYPOINTS['left_knee']),
        (KEYPOINTS['left_knee'], KEYPOINTS['left_ankle']),

        # Right side connections (green)
        (KEYPOINTS['right_shoulder'], KEYPOINTS['right_elbow']),
        (KEYPOINTS['right_elbow'], KEYPOINTS['right_wrist']),
        (KEYPOINTS['right_hip'], KEYPOINTS['right_knee']),
        (KEYPOINTS['right_knee'], KEYPOINTS['right_ankle']),

        # Shoulder to hip connections (blue)
        (KEYPOINTS['left_shoulder'], KEYPOINTS['left_hip']),
        (KEYPOINTS['right_shoulder'], KEYPOINTS['right_hip']),
        (KEYPOINTS['left_shoulder'], KEYPOINTS['right_shoulder']),
        (KEYPOINTS['left_hip'], KEYPOINTS['right_hip'])
    ]

    for keypoints in all_keypoints:
        for start, end in CONNECTIONS:
            if (0 <= keypoints[start][0] < image.shape[1] and
                    0 <= keypoints[start][1] < image.shape[0] and
                    0 <= keypoints[end][0] < image.shape[1] and
                    0 <= keypoints[end][1] < image.shape[0]):

                # Determine connection color based on the involved keypoints
                if start in [0, 1, 2, 3, 4] and end in [0, 1, 2, 3, 4]:  # Face connections
                    line_color = COLORS['face']
                elif start in [5, 7, 9, 11, 13, 15] and end in [7, 9, 13, 15]:
                    line_color = COLORS['left']
                elif start in [6, 8, 10, 12, 14, 16] and end in [8, 10, 14, 16]:
                    line_color = COLORS['right']
                elif (start in [5, 6, 11, 12] and end in [3, 4, 5, 6, 11, 12]):
                    line_color = COLORS['shoulder_hip']
                elif (start == KEYPOINTS['left_shoulder'] and end == KEYPOINTS['right_shoulder']) or \
                        (start == KEYPOINTS['right_shoulder'] and end == KEYPOINTS['left_shoulder']):
                    line_color = COLORS['shoulder_hip']
                elif (start == KEYPOINTS['left_hip'] and end == KEYPOINTS['right_hip']) or \
                        (start == KEYPOINTS['right_hip'] and end == KEYPOINTS['left_hip']):
                    line_color = COLORS['shoulder_hip']
                else:
                    line_color = (255, 255, 255)

                cv2.line(image, (int(keypoints[start][0]), int(keypoints[start][1])),
                         (int(keypoints[end][0]), int(keypoints[end][1])), line_color, line_size)

        for i in range(len(keypoints)):
            if keypoints[i][0] >= 0 and keypoints[i][1] >= 0:
                if i in [0, 1, 2, 3, 4]:
                    color = COLORS['face']
                elif i in [5, 7, 9, 11, 13, 15]:
                    color = COLORS['left']
                elif i in [6, 8, 10, 12, 14, 16]:
                    color = COLORS['right']
                else:
                    color = (255, 255, 255)

                cv2.circle(image, (int(keypoints[i][0]), int(keypoints[i][1])), point_size, color, -1)

    return image

@time_logger
def predict_keypoints(pose_model, img):
    pose_results = inference_topdown(pose_model, img)
    return pose_results[0].pred_instances.keypoints

@time_logger
def calculate_new_keypoints(pose_estimation_keypoints, saliency_coords, frame_shape):
    # Initialize an empty array with the same shape as keypoints
    new_keypoints = np.zeros_like(pose_estimation_keypoints)

    # Check if pred_keypoints[0] is not empty
    if pose_estimation_keypoints[0].size > 0:
        for i, keypoint in enumerate(pose_estimation_keypoints[0]):
            # Adjust keypoint coordinates with the given addition
            x, y = int(keypoint[0] + saliency_coords[0]), int(keypoint[1] + saliency_coords[1])
            new_keypoints[0][i] = [x, y]

            # Check if the new coordinates are within the image bounds
            if not (0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]):
                logging.critical(f"Keypoint {i} out of bounds: ({y}, {x})")
    else:
        logging.info("No keypoints found in the input array.")

    # Return the new keypoints array
    return new_keypoints

@time_logger
def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No overlap

    # Compute the area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both the bounding boxes
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2

    # Compute the intersection over union (IoU)
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

@time_logger
def predict_coordinate_with_ransac(last_10_frames, threshold=1, debug=False):
    """
    Predicts whether the current (10th) frame is an outlier based on the last 9 frames' X coordinates
    using RANSAC to identify outliers.

    Parameters:
    last_10_frames (list or np.ndarray): A list or array of the last 10 X coordinates.
    threshold (float): The maximum distance for a point to be classified as an inlier.
    debug (bool): If True, plots the result.

    Returns:
    tuple: A tuple containing:
        - predicted_coordinate (float): The predicted coordinate of the 10th frame.
        - is_outlier (bool): True if the current (10th) coordinate is considered an outlier.
    """

    # Ensure we have exactly 10 frames
    if len(last_10_frames) != 10:
        raise ValueError("Must provide exactly 10 frames for fitting.")

    # Convert to numpy array and exclude the last coordinate for fitting (only use the first 9)
    X = np.arange(9).reshape(-1, 1)  # Indices for the first 9 frames
    y = np.array(last_10_frames[:-1]).reshape(-1, 1)  # First 9 X coordinates

    ransac = RANSACRegressor(residual_threshold=threshold, random_state=42)

    # Fit the RANSAC model using the first 9 coordinates
    ransac.fit(X, y)

    # Predict the coordinate for the current (10th) frame
    predicted_coordinate = ransac.predict(np.array([[9]]))[0][0]

    # Check if the 10th frame is an outlier
    last_coordinate = last_10_frames[-1]
    inlier_mask = ransac.inlier_mask_  # Inliers from fitting the first 9 points
    residual = abs(last_coordinate - predicted_coordinate)
    is_outlier = residual > threshold

    return predicted_coordinate, is_outlier

outlier_detected = 0
@time_logger
def apply_ransac_to_keypoints(object_id, keypoint_index, keypoint_coord, axis, center, debug=False):
    """
    Applies RANSAC to detect and correct outliers in the keypoint coordinates (x or y).

    Parameters:
    - object_id (int): The ID of the tracked object.
    - keypoint_index (int): The index of the keypoint.
    - keypoint_coord (float): The current x or y coordinate of the keypoint.
    - axis (str): 'x' or 'y', indicating which coordinate is being processed.

    Returns:
    - corrected_coord (float): The corrected coordinate, either the original or the predicted one.
    """
    history = keypoints_history[object_id][f'{keypoint_index}_{axis}']

    if len(history) < 10:
        history.append(keypoint_coord)
        return keypoint_coord
    else:
        last_10_frames = history[-9:] + [keypoint_coord]
        predicted_coord, is_outlier = predict_coordinate_with_ransac(last_10_frames, debug=debug)
        history.append(keypoint_coord)

        if is_outlier:
            if center:
                distance_to_center = abs(keypoint_coord - center[0 if axis == 'x' else 1])
                correction_distance = abs(predicted_coord - keypoint_coord)
            return predicted_coord
        return keypoint_coord

@time_logger
def calculate_center(keypoints):
    valid_keypoints = [kp for kp in keypoints if kp[0] >= 0 and kp[1] >= 0]
    if not valid_keypoints:
        return None
    x_coords = [kp[0] for kp in valid_keypoints]
    y_coords = [kp[1] for kp in valid_keypoints]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return center_x, center_y

def adjust_keypoints(keypoints, x_offset=800, y_offset=0):
    """
    Adjust keypoint coordinates from a cropped image to the original image.

    Parameters:
    - keypoints: A list or NumPy array of keypoints, where each keypoint is a tuple (x, y)
    - x_offset: The x offset due to cropping (default: 800)
    - y_offset: The y offset due to cropping (default: 0, since the image was not cropped vertically)

    Returns:
    - adjusted_keypoints: A list of keypoints adjusted to the original image coordinates
    """
    adjusted_keypoints = []

    for x, y in keypoints:
        adjusted_x = x + x_offset
        adjusted_y = y + y_offset
        adjusted_keypoints.append((adjusted_x, adjusted_y))

    return np.array(adjusted_keypoints)

def cut_image(frame_end, frame_start, cut_dim):
    frame_start = frame_start[cut_dim[0]:cut_dim[1], cut_dim[2]:cut_dim[3]]
    frame_end = frame_end[cut_dim[0]:cut_dim[1], cut_dim[2]:cut_dim[3]]
    return frame_end, frame_start


@time_logger
def process_video(input_video_path, output_video_path, output_csv, pose_estimator, scale_percent=50, debug=False):
    """Process video frames and write visualized frames to a new video file."""
    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        cam_id = [item.split('cam')[1] for item in input_video_path.split('/')[-1].split('_') if item.startswith('cam')][0].split('.')[0]
        frame_start_path = f'start_frame_cam{cam_id}.jpg'
        print(f"Start frame path: {frame_start_path}")
        logging.info(f"Start frame path: {frame_start_path}")
        frame_start = cv2.imread(frame_start_path)
    except:
        logging.debug('No start frame found')

    #out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    frame_count = 0

    print("Processing Video Frames")
    print(f'Total Frames: {total_frames}')

    tracked_objects = []  # Stores object IDs and their bounding boxes
    next_object_id = 0  # Initial ID for the next object
    iou_threshold = 0.3  # IoU threshold to decide if it’s the same object
    keypoints = []  # List to store all keypoints

    scale_factor = int(100 / scale_percent)  # Calculate the scale factor

    data = []
    with tqdm(total=total_frames, desc="Processing Video Frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cut_dim = (0, 2000, 800, 2700)
            start_frame = frame_start.copy()

            frame, start_frame = cut_image(frame, start_frame, cut_dim)

            # Pre-process frames
            frame_diff, frame_end = process_frames(start_frame, frame, scale_percent=scale_percent, debug=debug)

            # Detect bounding boxes
            bounding_boxes = saliency_bounding_boxes(frame_diff, debug=debug)
            cropped_images, coordinates = crop_image_to_bounding_boxes(frame_end, bounding_boxes, debug=debug)

            all_keypoints = []  # List to store all keypoints
            current_frame_objects = []  # Store objects detected in this frame

            for idx, (cropped_image, (x_offset, y_offset)) in enumerate(zip(cropped_images, coordinates)):
                predicted_keypoints = predict_keypoints(pose_estimator, cropped_image)
                if len(predicted_keypoints) > 0:
                    # Adjust keypoints to the original image coordinates
                    new_keypoints = calculate_new_keypoints(predicted_keypoints, (x_offset, y_offset), frame.shape)
                    all_keypoints.append(new_keypoints)

                    # Scale bounding box and keypoints to original size
                    scaled_bbox = [coord * scale_factor for coord in bounding_boxes]
                    scaled_keypoints = new_keypoints[0] * scale_factor
                    adjusted_keypoints = np.copy(scaled_keypoints)
                    adjusted_keypoints[:, 0] += cut_dim[2]
                    adjusted_keypoints[:, 1] += cut_dim[0]


                    # Append data to the list
                    data.append({
                        'frame': frame_count,
                        #'object_id': object_id,
                        'bbox': scaled_bbox,
                        'keypoints': adjusted_keypoints.tolist()
                    })

            # Update tracked objects with the current frame's detected objects
            tracked_objects = current_frame_objects

            # Flatten the list of new keypoints
            keypoints_in_original_image = np.concatenate(all_keypoints, axis=0) if all_keypoints else []

            # Upscale the frame and keypoints back to the original size
            frame_end = cv2.resize(frame_end, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            keypoints_in_original_image *= scale_factor

            # Visualize keypoints on the frame
            #vis_frame_bgr = visualize_coco_keypoints(frame_end, keypoints_in_original_image, point_size=5, line_size=2)
            #out.write(vis_frame_bgr)

            pbar.update(1)
            frame_count += 1

    cap.release()
    #out.release()

    print(f"Number of outliers detected: {outlier_detected}")

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'bbox', 'keypoints'] # 'object_id',
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f"Video processing complete. Saved to {output_video_path}")
    if debug:
        with open('keypoints.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Frame', 'Object', 'KP_ID', 'X', 'Y']  # Include object index and keypoint ID
            writer.writerow(header)

            # Write keypoints for each frame and object
            for frame_idx, frame_keypoints in enumerate(keypoints):
                for obj_idx, obj_keypoints in enumerate(frame_keypoints):
                    if obj_keypoints.shape[0] > 0:  # Ensure there are keypoints for this object
                        for kp_id in range(obj_keypoints.shape[0]):
                            # Write the frame number, object index, keypoint ID, and x/y coordinates
                            x = obj_keypoints[kp_id][0]
                            y = obj_keypoints[kp_id][1]
                            writer.writerow([frame_idx, obj_idx + 1, kp_id + 1, x, y])

def main(pose_config, pose_checkpoint, input_video_path, output_video_path, output_csv, log_level, debug=False):
    logging.basicConfig(level=log_level)

    wandb.init(project="climbing-metrics", config={"debug": True})

    pose_model = init_pose_estimation_model(pose_config, pose_checkpoint, device)
    process_video(input_video_path, output_video_path, output_csv, pose_model, debug=debug)

    print("Duration Summary:")
    print(durations.keys())
    print(f"Total Duration: {sum(durations['process_video']):.2f} seconds")
    print(f'Average duration for process_frames: {sum(durations["process_frames"]) / len(durations["process_frames"]):.2f} seconds')
    print(f'Average duration for saliency_bounding_boxes: {sum(durations["saliency_bounding_boxes"]) / len(durations["saliency_bounding_boxes"]):.2f} seconds')
    print(f'Average duration for predict_keypoints: {sum(durations["predict_keypoints"]) / len(durations["predict_keypoints"]):.2f} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a video for pose estimation and logging.")

    # Adding arguments for pose configuration and checkpoint
    parser.add_argument('--pose_config', type=str, required=True,
                        help='Path to the pose estimation configuration file.')
    parser.add_argument('--pose_checkpoint', type=str, required=True,
                        help='Path to the pose estimation model checkpoint.')

    # Adding arguments for input and output video paths
    parser.add_argument('--input_video_path', type=str, required=True,
                        help='Path to the input video file.')
    parser.add_argument('--output_video_path', type=str, required=True,
                        help='Path for saving the output video file.')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Path for saving the output video file.')

    # Adding argument for logging level
    parser.add_argument('--log_level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set the logging level.')

    args = parser.parse_args()

    log_level = getattr(logging, args.log_level)

    if (log_level != logging.ERROR) and (log_level != logging.CRITICAL):
        debug = True
    else:
        debug = False

    # Call the main function with parsed arguments
    main(args.pose_config, args.pose_checkpoint, args.input_video_path, args.output_video_path, args.output_csv, log_level, debug=False)
    