import cv2
from PIL import Image
import random
import argparse
import os
import os.path as osp
import mediapipe as mp
import numpy as np
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
import isl_utils as islutils

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Visualize random video')
    parser.add_argument('--video_dir', default=islutils.VIDEO_DIR, type=str, help='Directory containing videos')
    args = parser.parse_args()

    # Initialize video capture
    vid = random.choice(list(islutils.allfiles(args.video_dir)))
    print('Running pose inference ...', vid)
    cap = cv2.VideoCapture(vid)
    print('Video has fps ...', cap.get(cv2.CAP_PROP_FPS))
    print('Video has frame count ...', cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Video has duration (s) ...', cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    print('Video has cut scenes ...', len(detect(vid, AdaptiveDetector())))

    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imshow(f'Annotated Video | {vid}', islutils.aspectRatioPreservingResize(image, 512))

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    cap.release()
    cv2.destroyAllWindows()
