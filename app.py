from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
app = Flask(__name__, static_url_path='/static')

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connect

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Videos worth of data
no_sequences = 70

# Videos are going to be 23 frames in length
sequence_length = 23

# Function to get the next available sequence number in a folder
def get_next_sequence_number(folder):
    full_path = os.path.join(DATA_PATH, folder)
    if not os.path.exists(full_path):
        return 1  # Start with 1 if the folder does not exist yet
    sequences = [int(d) for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]
    if sequences:
        return max(sequences) + 1
    else:
        return 1  # Start with 1 if there are no sequences in the folder

# Function to add frames to the video sequence
def add_frames_to_sequence(folder, no_sequences):
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.2) as holistic:
        # Get the next available sequence number
        start_sequence_number = get_next_sequence_number(folder)
        # Loop through video length aka sequence length
        for sequence in range(start_sequence_number, start_sequence_number + no_sequences):
            remaining_sequences = no_sequences - (sequence - start_sequence_number)
            for frame_num in range(sequence_length):
                # Read feed
                ret, frame = cap.read()
                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # Draw landmarks
                draw_styled_landmarks(image, results)
                # Print sequence number
                cv2.putText(image, f'({remaining_sequences} left)', (15, 40),  
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2, cv2.LINE_AA)
                # Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    # Show to screen
                    cv2.namedWindow('OpenCV Feed', cv2.WINDOW_NORMAL)  # Create window with normal size
                    cv2.resizeWindow('OpenCV Feed', 800, 600)  # Resize window
                    cv2.imshow('OpenCV Feed', image)
                    cv2.setWindowProperty('OpenCV Feed', cv2.WND_PROP_TOPMOST, 1)  # Set window to stay on top
                    cv2.waitKey(1000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(folder, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.namedWindow('OpenCV Feed', cv2.WINDOW_NORMAL)  # Create window with normal size
                    cv2.resizeWindow('OpenCV Feed', 800, 600)  # Resize window
                    cv2.imshow('OpenCV Feed', image)
                    cv2.setWindowProperty('OpenCV Feed', cv2.WND_PROP_TOPMOST, 1)  # Set window to stay on top
                
                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_dir = os.path.join(DATA_PATH, folder, str(sequence))
                os.makedirs(npy_dir, exist_ok=True)  # Create directories if they do not exist
                npy_path = os.path.join(npy_dir, str(frame_num) + '.npy')
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_query = request.form.get('search')
        if search_query:
            filtered_files = [file for file in os.listdir('MP_Data') if search_query.lower() in file.lower()]
            return render_template('index.html', files=filtered_files, current_path='', search_query=search_query)
    files = sorted(os.listdir('MP_Data'), key=lambda x: int(x) if x.isdigit() else x, reverse=True)
    return render_template('index.html', files=files, current_path='', search_query='')

@app.route('/browse', methods=['POST'])
def browse():
    folder = request.form.get('folder')
    current_path = request.form.get('current_path', '')
    full_path = os.path.join('MP_Data', current_path, folder)
    action = request.form.get('action')  # Get the value of the action parameter
    
    if action == 'add_frames':
        no_sequences = int(request.form.get('no_sequences', 70))  # Get the value of no_sequences input field
        add_frames_to_sequence(folder, no_sequences)  # Call a function to add frames to the sequence
        return f'Frames added to sequence in folder: {folder}'

    if os.path.isdir(full_path):
        files = sorted(os.listdir(full_path), key=lambda x: int(x) if x.isdigit() else x, reverse=True)
        return render_template('index.html', files=files, current_path=os.path.join(current_path, folder))
    else:
        return f'{folder} is not a directory'

@app.route('/create_folder', methods=['POST'])
def create_folder():
    new_folder_name = request.form.get('new_folder')  # Get the new folder name from the form
    current_path = request.form.get('current_path', '')  # Optionally, you might want to get the current path
    full_path = os.path.join('MP_Data', current_path, new_folder_name)  # Construct the full path for the new folder
    
    # Check if the folder already exists
    if os.path.exists(full_path):
        return redirect('/')
    
    # Create the new folder
    try:
        os.makedirs(full_path)
        return redirect('/')
    except Exception as e:
        return f'Failed to create folder "{new_folder_name}": {str(e)}'
    
    
import shutil

@app.route('/delete_folder', methods=['POST'])
def delete_folder():
    folder_name = request.form.get('folder')  # Get the folder name from the form
    current_path = request.form.get('current_path', '')  # Optionally, you might want to get the current path
    full_path = os.path.join('MP_Data', current_path, folder_name)  # Construct the full path for the folder
    # Check if the folder exists
    if os.path.exists(full_path) and os.path.isdir(full_path):
        try:
            shutil.rmtree(full_path)  # Remove the directory and its contents
            return redirect('/')  # Redirect to the home page after successful deletion
        except Exception as e:
            return redirect('/')
    else:
        return f'Folder "{folder_name}" does not exist or is not a directory.'

if __name__ == '__main__':
    app.run(debug=True)