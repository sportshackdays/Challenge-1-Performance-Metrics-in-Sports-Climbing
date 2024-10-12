import os

import pandas as pd
import numpy as np
import ast
import cv2
from tqdm import tqdm

def get_bounding_box_from_keypoints(row):
    """
    This function takes a row of a DataFrame where keypoints are stored as 'keypoint_x' and 'keypoint_y' pairs,
    finds the minimum and maximum coordinates, and returns the bounding box as (x, y, w, h).

    Parameters:
    row (pandas.Series): A row of a DataFrame containing keypoints for a frame.

    Returns:
    bbox (tuple): A tuple (x, y, w, h) representing the bounding box.
    """
    # Filter out all keypoint_x and keypoint_y columns from the row
    keypoint_x_cols = [col for col in row.index if '_x' in col]
    keypoint_y_cols = [col for col in row.index if '_y' in col]

    # Extract the x and y coordinates as lists
    keypoint_x_values = row[keypoint_x_cols].values
    keypoint_y_values = row[keypoint_y_cols].values

    # Find the minimum and maximum values of x and y
    min_x, max_x = min(keypoint_x_values), max(keypoint_x_values)
    min_y, max_y = min(keypoint_y_values), max(keypoint_y_values)

    # Calculate the width and height of the bounding box
    width = max_x - min_x + 20
    height = max_y - min_y + 20

    # Return the bounding box as (x, y, w, h)
    return pd.Series([int(min_x), int(min_y), int(width), int(height)], index=['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])


def convert_string_to_array(string):
    return np.array(ast.literal_eval(string))


def is_bbox_overlap(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    x2_min, y2_min, x2_max, y2_max = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]

    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

def remove_duplicates(bboxes):
    unique_bboxes = []
    seen = set()  # Set to keep track of seen bounding boxes

    for bbox in bboxes:
        # Create a tuple for the first four elements only
        simplified_bbox = tuple(bbox[:4])
        if simplified_bbox not in seen:
            seen.add(simplified_bbox)  # Add to seen set
            unique_bboxes.append(simplified_bbox)  # Add to the unique list

    return unique_bboxes

def extract_keypoints(df_cam14):
    
    """
    Extracts keypoints from the 'keypoints' column of the dataframe and creates new columns for each keypoint's x and y coordinates.

    Parameters:
    df_cam14 (pd.DataFrame): DataFrame containing a 'keypoints' column with keypoint data.

    Returns:
    pd.DataFrame: DataFrame with new columns for each keypoint's x and y coordinates.
    """
    for i in range(17):
        df_cam14[f'keypoint_{i+1:02d}_x'] = df_cam14['keypoints'].apply(lambda k: k[i][0])
        df_cam14[f'keypoint_{i+1:02d}_y'] = df_cam14['keypoints'].apply(lambda k: k[i][1])
    
    df_cam14.to_csv('./output/extracted_keypoints.csv')
    return df_cam14

def assign_object_ids(df):
    df['object_id'] = 0
    for i in range(len(df)):
        current_bbox = df.loc[i, ['bbox_keypoint_x', 'bbox_keypoint_y', 'bbox_keypoint_w', 'bbox_keypoint_h']].values
        found_overlap = False

        # Check the previous 20 rows for overlapping bounding boxes
        for j in range(max(0, i - 5), i):
            previous_bbox = df.loc[j, ['bbox_keypoint_x', 'bbox_keypoint_y', 'bbox_keypoint_w', 'bbox_keypoint_h']].values
            if is_bbox_overlap(current_bbox, previous_bbox):
                df.loc[i, 'object_id'] = df.loc[j, 'object_id']
                found_overlap = True
                break

        # If no overlap is found, assign a new object_id
        if not found_overlap:
            df.loc[i, 'object_id'] = df['object_id'].max() + 1

    return df

def expand_keypoints(row):
    keypoints = row['keypoints_array']

    # Check if the shape is (x, 17, 2)
    if len(keypoints.shape) == 3 and keypoints.shape[1:] == (17, 2):
        # If shape is (1, 17, 2), reshape to (17, 2) and return as a single row
        if keypoints.shape[0] == 1:
            # Return a single row, keep all other columns
            row_copy = row.copy()
            row_copy['keypoints_array'] = keypoints[0]  # Flatten to (17, 2)
            return pd.DataFrame([row_copy])

        # If shape is (x, 17, 2), create x new rows with the same values for other columns
        expanded_rows = pd.DataFrame([row] * keypoints.shape[0])  # Replicate the original row
        expanded_rows['keypoints_array'] = list(keypoints)  # Replace with split (17, 2) arrays
        return expanded_rows
    else:
        # If the shape is already (17, 2), return the row unchanged as a DataFrame
        return pd.DataFrame([row])

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


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    return cap

# csv_path = '../output/keypoints/semifinals_lead_women_and_men_cam14_03_pose_estimation_data.csv'

def load_keypoints(csv_path):
    df = pd.read_csv(csv_path)
    return preprocess_dataframe(df)


def plot_keypoints_on_frame(frame, keypoints, frame_number):
    keypoints_all = []
    for _, row in keypoints[keypoints['frame'] == frame_number].iterrows():
        keypoints_object = []
        for i in range(17):
            x = int(row[f'keypoint_{i + 1:02d}_x'])
            y = int(row[f'keypoint_{i + 1:02d}_y'])
            keypoints_object.append((x, y))
        keypoints_all.append(keypoints_object)
    frame = visualize_coco_keypoints(frame, keypoints_all)
    return frame


def save_video(frames, output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    for frame in frames:
        out.write(frame)
    out.release()


def filter_object_ids(df, min_frames=100):
    """
    Drop all object_id with less than min_frames frames.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    min_frames (int): Minimum number of frames an object_id must have to be retained.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    return df[df['object_id'].map(df['object_id'].value_counts()) > min_frames]

def smooth_keypoints(df_cam14_filtered, window_size=5):
    df_cam14_smooth = df_cam14_filtered.copy()
    for object_id in df_cam14_filtered['object_id'].unique():
        df_object = df_cam14_filtered[df_cam14_filtered['object_id'] == object_id]
        for i in range(window_size, len(df_object) - window_size):
            for j in range(17):
                df_cam14_smooth.loc[df_object.index[i], f'keypoint_{j+1:02d}_x'] = df_object[f'keypoint_{j+1:02d}_x'].iloc[i-5:i+5].mean()
                df_cam14_smooth.loc[df_object.index[i], f'keypoint_{j+1:02d}_y'] = df_object[f'keypoint_{j+1:02d}_y'].iloc[i-5:i+5].mean()
    return df_cam14_smooth

def preprocess_dataframe(df):
    # Convert the 'keypoints' column from string to list
    df['keypoints_array'] = df['keypoints'].apply(convert_string_to_array)
    df = pd.concat(df.apply(expand_keypoints, axis=1).reset_index(drop=True).tolist(), ignore_index=True)
    df.drop(['keypoints', 'bbox'], axis=1, inplace=True)
    df.rename(columns={'keypoints_array': 'keypoints'}, inplace=True)
    df = extract_keypoints(df)
    df[['bbox_keypoint_x','bbox_keypoint_y','bbox_keypoint_w','bbox_keypoint_h']] = df.apply(get_bounding_box_from_keypoints, axis=1)
    df = assign_object_ids(df)
    df = filter_object_ids(df, min_frames=100)
    df_smooth = smooth_keypoints(df)
    # df_smooth.to_csv('output/smoothing
    return df_smooth #


def process_video(video_path, keypoints, output_path):
    cap = load_video(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    frame_number = 0
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = plot_keypoints_on_frame(frame, keypoints, frame_number)
            out.write(frame)
            frame_number += 1
            pbar.update(1)

    cap.release()
    out.release()
    print("Video saved successfully.")

# video_path, csv_path, output_video_path
# video_path = '../data/videos/semifinals_lead_women_and_men_cam14_03.avi'
# csv_path = '../output/keypoints/semifinals_lead_women_and_men_cam14_03_pose_estimation_data.csv'
# output_video_path = './output/smoothing/semifinals_lead_women_and_men_cam14_03_smoothed.mp4'

def main(video_path, csv_path, output_video_path):
    df_keypoints = load_keypoints(csv_path)
    process_video(video_path, df_keypoints, output_video_path)


if __name__ == "__main__":
    csv_folder = '../output/keypoints'
    video_folder = '../data/videos'
    video_output_folder = '../output/smoothing'
    for filename in os.listdir(csv_folder):
        if filename.endswith('.csv'):
            video_name = filename.split('_pose_estimation_data.csv')[0]
            csv_path = os.path.join(csv_folder, filename)
            video_path = os.path.join(video_folder, video_name + '.avi')
            output_video_path = os.path.join(video_output_folder, video_name + '_smoothed.mp4')
            if not os.path.exists(output_video_path):
                main(video_path, csv_path, output_video_path)

# video_path, csv_path, output_video_path
# video_path = '../data/videos/semifinals_lead_women_and_men_cam14_03.avi'
# csv_path = '../output/keypoints/semifinals_lead_women_and_men_cam14_03_pose_estimation_data.csv'
# output_video_path = './output/smoothing/semifinals_lead_women_and_men_cam14_03_smoothed.mp4'
