import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO messages

import cv2
import mediapipe as mp
import pandas as pd

# Initialize Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_video(video_path, output_csv, frame_rate=1):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frame_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Extract frames at the specified frame_rate
        if frame_count % frame_rate != 0:
            continue
        
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Directly append the frame to the frame_data
        frame_data.append(image_rgb)

    # Release the video capture object
    cap.release()

    # Process each frame with Mediapipe pose estimation
    pose_results = []
    for frame in frame_data:
        results = pose.process(frame)
        
        if results.pose_landmarks:
            frame_landmarks = {}
            for id, lm in enumerate(results.pose_landmarks.landmark):
                frame_landmarks[f'x{id}'] = lm.x
                frame_landmarks[f'y{id}'] = lm.y
                frame_landmarks[f'z{id}'] = lm.z
                frame_landmarks[f'v{id}'] = lm.visibility
            pose_results.append(frame_landmarks)

    # Save the pose results to CSV
    df = pd.DataFrame(pose_results)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Append to the CSV file if it exists, otherwise create a new one
    if os.path.exists(output_csv):
        df.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        df.to_csv(output_csv, index=False)

def main():
    print("GET INTO MAIN CLASS")
    base_dir = '../Video_Dataset'
    new_dir = "CSV_Dataset"
    
    count_folder = 0
    for folder_name in os.listdir(base_dir):
        print("THIS IS COUNT FOLDER: ", count_folder)
        folder_path = os.path.join(base_dir, folder_name)
        
        if os.path.isdir(folder_path):
            folder_path = os.path.join(folder_path, os.listdir(folder_path)[0])
            print("THIS IS THE FOLDER PATH:", folder_path)
            for video_name in os.listdir(folder_path):
                print("THIS IS THE VIDEO NAME", video_name)
                video_path = os.path.join(folder_path, video_name)
                
                if video_path.endswith('.mp4'):
                    # Construct the output CSV path
                    output_csv = os.path.join(new_dir, folder_name +'.csv')
                    
                    print(f'Processing {video_path} and saving to {output_csv}')
                    process_video(video_path, output_csv, frame_rate=1)  # Adjust frame_rate as needed

        count_folder += 1

if __name__ == "__main__":
    main()


# import os
# import cv2
# import mediapipe as mp
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor

# # Suppress TensorFlow INFO messages
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Initialize Mediapipe
# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

# def process_video(video_path, output_csv, frame_rate=1):
#     cap = cv2.VideoCapture(video_path)
#     frame_data = []
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_count += 1
        
#         if frame_count % frame_rate != 0:
#             continue
        
#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_data.append(image_rgb)

#     cap.release()

#     # Process frames in batches for efficiency
#     pose_results = []
#     batch_size = 10
#     for i in range(0, len(frame_data), batch_size):
#         batch_frames = frame_data[i:i+batch_size]
#         batch_results = pose.process_many(batch_frames)
        
#         for results in batch_results:
#             if results.pose_landmarks:
#                 frame_landmarks = {}
#                 for id, lm in enumerate(results.pose_landmarks.landmark):
#                     frame_landmarks[f'x{id}'] = lm.x
#                     frame_landmarks[f'y{id}'] = lm.y
#                     frame_landmarks[f'z{id}'] = lm.z
#                     frame_landmarks[f'v{id}'] = lm.visibility
#                 pose_results.append(frame_landmarks)

#     df = pd.DataFrame(pose_results)

#     os.makedirs(os.path.dirname(output_csv), exist_ok=True)
#     df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)

# def process_video_wrapper(args):
#     video_path, output_csv = args
#     process_video(video_path, output_csv, frame_rate=1)

# def main():
#     base_dir = '../Video_Dataset'
#     new_dir = "CSV_Dataset"

#     with ThreadPoolExecutor() as executor:
#         for folder_name in os.listdir(base_dir):
#             folder_path = os.path.join(base_dir, folder_name)
            
#             if os.path.isdir(folder_path):
#                 folder_path = os.path.join(folder_path, os.listdir(folder_path)[0])
#                 for video_name in os.listdir(folder_path):
#                     video_path = os.path.join(folder_path, video_name)
                    
#                     if video_path.endswith('.mp4'):
#                         output_csv = os.path.join(new_dir, folder_name + '.csv')
#                         print(f'Processing {video_path} and saving to {output_csv}')
                        
#                         # Submitting the task to ThreadPoolExecutor
#                         executor.submit(process_video_wrapper, (video_path, output_csv))

# if __name__ == "__main__":
#     main()
