import cv2
import argparse
import os
import os.path as osp
import random
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import requests
from PIL import Image
import isl_utils as islutils


def draw_landmarks_on_image(rgb_image, detection_result):
    """ Visualization code that I stole from Github """
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Visualize random video')
    parser.add_argument('--video_dir', type=str, help='Directory containing videos')
    parser.add_argument('--smaller_dim', type=int, help='Size of rendered visualization')
    args = parser.parse_args()

    # Download the pose model
    if not osp.exists('pose_landmarker.task') :
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
        response = requests.get(url)

        with open('pose_landmarker.task', 'wb') as f:
            f.write(response.content)

    # Set up the pose detector
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # Initialize video capture
    vid = random.choice(list(islutils.allfiles(args.video_dir)))
    print('Running pose inference ...', vid)
    cap = cv2.VideoCapture(vid)

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Video stream ended or file not found.")
            break

        # Display the frame
        detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
        annotated_image = draw_landmarks_on_image(frame, detection_result)
        resized_image = islutils.aspectRatioPreservingResize(annotated_image, args.smaller_dim)
        cv2.imshow(f'Annotated Video | {vid}', resized_image) 
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

