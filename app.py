# --------------------------libraries----------------------------------------------------------------------------------
import os
import openpyxl
import cap
import time
import joblib
import scipy
import sqlite3
import pandas as pd
import cv2
import numpy as np
import shutil
import tensorflow as tf
from tqdm import tqdm
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, Response
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import face_recognition
import pickle
#----------------------------------------------------------------------------------------------------------------------

# Flask App Initialization
app = Flask(__name__)
app.secret_key = 'your_secret_key'
model_generation_time = None
#-------------------------------------folders--------------------------------------------------------------------------
# Directory Configurations
dataset_path = 'datasets/CSM/csm_images'
train_path = 'datasets/CSM/csmp/train'
test_path = 'datasets/CSM/csmp/test'
val_path = 'datasets/CSM/csmp/val'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
EXTRACT_FRAMES_FOLDER = 'extract_frames/'
DETECT_FOLDER = 'detect_objects/'
#----------------------------------------------------------------------------------------------------------------------

#----------------------------------detection objects-------------------------------------------------------------------
# Ensure the detect_objects folder exists
if not os.path.exists(DETECT_FOLDER):
    os.makedirs(DETECT_FOLDER)

# Function to detect faces in a single frame
def detect_faces_in_single_frame():
    #Clear the detect_objects folder
    for file in os.listdir(DETECT_FOLDER):
        file_path = os.path.join(DETECT_FOLDER, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Select one frame (e.g., the first frame in the extract_frames folder)
    frames = sorted(os.listdir(EXTRACT_FRAMES_FOLDER))
    if not frames:
        print("No frames found in the extract_frames folder.")
        return "No frames found."

    selected_frame_path = os.path.join(EXTRACT_FRAMES_FOLDER, frames[1])  # First frame
    print(f"Selected frame: {selected_frame_path}")

    # Read the selected frame
    frame = cv2.imread(selected_frame_path)
    if frame is None:
        print("Failed to read the selected frame.")
        return "Failed to read frame."

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Number of faces detected: {len(faces)}")

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Save the processed frame to the detect_objects folder
    output_frame_path = os.path.join(DETECT_FOLDER, 'detected_faces.jpg')
    cv2.imwrite(output_frame_path, frame)
    print(f"Processed frame saved to {output_frame_path}.")

    # Return the count of detected faces
    return {"message": "Face detection complete.", "face_count": len(faces), "frame_path": output_frame_path}
#----------------------------------------------------------------------------------------------------------------------

#---------------------------------------------uploading videos---------------------------------------------------------
# Ensure train, test, val directories exist
for path in [train_path, test_path, val_path]:
    os.makedirs(path, exist_ok=True)

# Allowed File Types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Upload Video Route
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video_file' not in request.files:
        return jsonify({"success": False, "message": "No file part."}), 400

    file = request.files['video_file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file."}), 400

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_video.mp4')
        file.save(file_path)

        # Path to the pre-trained model and student images
        model_path = 'face_encodings.pkl'
        student_images_path = 'datasets/CSM/csm_images'

        # Train the model only if the .pkl file doesn't exist
        if not os.path.exists(model_path):
            train_face_recognition_model(student_images_path, model_path)

        extract_and_replace_frames(file_path)

        # Detect faces in a frame
        face_detection_result = detect_faces_in_single_frame()
        detected_frame_path = os.path.join(DETECT_FOLDER, 'detected_faces.jpg')

        if not face_detection_result["frame_path"]:
            return jsonify({
                "success": False,
                "message": "Failed to process frames or save the detected frame."
            })

        # Recognize students' roll numbers
        recognized_roll_numbers = recognize_faces(face_detection_result["frame_path"], model_path)

        return jsonify({"success": True, "message": "Video uploaded and frames replaced.", "face_detection_result": face_detection_result["message"], "face_count": face_detection_result["face_count"],"processed_frame_path": face_detection_result["frame_path"], "recognized_students": recognized_roll_numbers})
    else:
        return jsonify({"success": False, "message": "Invalid file type."}), 400
#----------------------------------------------------------------------------------------------------------------------

def train_face_recognition_model(student_images_path, model_path):
    # Check if the model already exists
    if os.path.exists(model_path):
        print("Model file already exists. Skipping training.")
        return

    face_encodings = []
    roll_numbers = []

    # Get a list of student folders
    student_folders = [folder for folder in os.listdir(student_images_path) if os.path.isdir(os.path.join(student_images_path, folder))]
    print(f"Training on {len(student_folders)} students' data...")

    # Iterate through each student's folder
    for student_folder in tqdm(student_folders, desc="Processing Students", unit="student"):
        student_folder_path = os.path.join(student_images_path, student_folder)

        # if not os.path.isdir(student_folder_path):
        #     continue  # Skip if it's not a folder

        # Each folder contains images of a single student
        for image_file in os.listdir(student_folder_path):
            image_path = os.path.join(student_folder_path, image_file)

            # Load the image
            image = face_recognition.load_image_file(image_path)

            # Get face encodings
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:  # Ensure a face was detected
                face_encodings.append(encodings[0])
                roll_numbers.append(student_folder)  # Use the folder name (roll number) as the label

    # Save encodings and roll numbers to a .pkl file
    with open(model_path, 'wb') as f:
        pickle.dump({"encodings": face_encodings, "roll_numbers": roll_numbers}, f)

    print(f"Model saved to {model_path}")

def recognize_faces(detected_frame_path, model_path):
    # Load the model (encodings and roll numbers)
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_roll_numbers = data["roll_numbers"]

    # Load the frame with detected faces
    image = face_recognition.load_image_file(detected_frame_path)

    # Get face encodings for faces in the frame
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    recognized_roll_numbers = []

    # Compare each face in the frame with known encodings
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        if True in matches:
            match_index = matches.index(True)
            recognized_roll_numbers.append(known_roll_numbers[match_index])

    return recognized_roll_numbers



#----------------------------------------extract frames----------------------------------------------------------------
# Extract and Replace Frames
def extract_and_replace_frames(video_path):

    # Directory to store frames
    output_frames_dir = 'extract_frames/'
    os.makedirs(output_frames_dir, exist_ok=True)

    # Clear existing frames
    for frame_file in os.listdir(output_frames_dir):
        frame_path = os.path.join(output_frames_dir, frame_file)
        os.remove(frame_path)

    # Extract frames from the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while frame_count<5:
        ret, frame = cap.read()
        if not ret:
            break

        # Save each frame with a unique name
        frame_path = os.path.join(output_frames_dir, f'frame_{frame_count + 1}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"{frame_count} frames extracted and replaced in {output_frames_dir}.")
#----------------------------------------------------------------------------------------------------------------------

#------------------------------------------preprocessing images--------------------------------------------------------
# Preprocess Images
def preprocess_images():
    faces = []
    labels = []

    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces_in_image = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces_in_image:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (100, 100))
                faces.append(face)
                labels.append(int(filename.split('.')[0]))

    faces = np.array(faces) / 255.0
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    for i in range(len(X_train)):
        cv2.imwrite(os.path.join(train_path, f'{y_train[i]}_{i}.jpg'), X_train[i] * 255)
    for i in range(len(X_test)):
        cv2.imwrite(os.path.join(test_path, f'{y_test[i]}_{i}.jpg'), X_test[i] * 255)
    for i in range(len(X_val)):
        cv2.imwrite(os.path.join(val_path, f'{y_val[i]}_{i}.jpg'), X_val[i] * 255)

    return X_train, X_val, X_test, y_train, y_val, y_test
#----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------build model------------------------------------------------------------------
# Build Model
def build_model(input_shape=(100, 100, 1), num_classes=10):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
#----------------------------------------------------------------------------------------------------------------------

#-------------------------------------------------train model----------------------------------------------------------
# Train Model
def train_model(X_train, X_val, y_train, y_val, num_classes):
    model = build_model(num_classes=num_classes)
    X_train = X_train.reshape(-1, 100, 100, 1)
    X_val = X_val.reshape(-1, 100, 100, 1)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    model.save("datasets/CSM/csmt/csm.h5")
#----------------------------------------------------------------------------------------------------------------------

#---------------------------------------recognize face-----------------------------------------------------------------
# Recognize Face
def recognize_face(frame, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))
        face_resized = face_resized.reshape(1, 100, 100, 1) / 255.0
        prediction = model.predict(face_resized)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("Face Recognition", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#----------------------------------------------------------------------------------------------------------------------

#---------------------------------through cc cam-----------------------------------------------------------------------
# Route to access CCTV camera
@app.route('/access_cctv', methods=['POST'])
def access_cctv():
    data = request.get_json()
    room_id = data.get('room_id')

    # Replace with actual RTSP URL for the CCTV in the room
    rtsp_url = f"rtsp://username:password@cctv-ip-address/{room_id}"

    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return jsonify({"success": False, "message": "Failed to connect to CCTV."})

        # Process video stream (example: capture a frame)
        ret, frame = cap.read()
        if ret:
            # Save the frame for further processing
            cv2.imwrite('cctv_frame.jpg', frame)
        cap.release()
        return jsonify({"success": True, "message": "CCTV access successful."})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
#----------------------------------------------------------------------------------------------------------------------

#----------------------------------through mobile cam------------------------------------------------------------------
# Route to get mobile camera link
@app.route('/get_mobile_link', methods=['POST'])
def get_mobile_link():
    data = request.get_json()
    imei = data.get('imei')

    # Simulate linking mobile camera
    # Example: Use a custom mobile app or WebRTC to get a camera feed URL
    if imei == '861612055290225':  # Replace with real IMEI validation
        mobile_camera_url = "http://192.168.1.3:8080/video"  # Replace with dynamic IP
    else:
        return jsonify({"success": False, "message": "Invalid IMEI."})

    # Open a connection to the IP Webcam video stream
    cap = cv2.VideoCapture(mobile_camera_url)

    # Define the duration for the video recording (in seconds)
    # Record video from the IP Webcam feed
    output_file = "output_video.mp4"
    record_duration = 10  # Record duration in seconds
    fps = 20
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a named window with normal size (no auto-resizing)
    cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)

    # Resize window to match the exact size of the video capture frame
    cv2.resizeWindow("Recording", frame_width, frame_height)

    # Check if the connection is successful
    if not cap.isOpened():
        print("Error: Unable to access the video stream!")
        return jsonify({"success": False, "message": "Unable to access the video stream."})

    # Define the codec and create a VideoWriter object
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))

    print("Recording started...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            print("Error: Unable to read from the video stream!")
            break

        # Write the frame to the output file
        out.write(frame)

        # Display the frame in a pop-up window (this is the live video window)
        cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Add timestamp to frame
        cv2.imshow("Recording", frame)  # Create a window to display the video feed

        # Stop recording after the specified duration
        if time.time() - start_time > record_duration:
            print("Recording finished!")
            break

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Extract frames after recording
    extracted_frames = extract_frames(output_file, num_frames=5)

    return jsonify({
        "success": True,
        "message": "Video recorded and frames extracted successfully.",
        "extracted_frames": extracted_frames
    })
#----------------------------------------------------------------------------------------------------------------------

#------------------------------------extract frames--------------------------------------------------------------------
# Folder where the frames will be stored
frames_folder = "extract_frames"

# Ensure the folder exists
if not os.path.exists(frames_folder):
    os.makedirs(frames_folder)

# Function to extract frames from the video
def extract_frames(video_path, num_frames=5):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval between frames to extract
    frame_interval = total_frames // num_frames

    frame_paths = []  # Store paths of the saved frames

    # Extract frames at equal intervals
    for i in range(num_frames):
        # Set the current frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()

        if ret:
            # Save the frame with a unique name (replace previous frames)
            frame_filename = os.path.join(frames_folder, f"frame_{i + 1}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_paths.append(frame_filename)

    cap.release()
    return frame_paths
#----------------------------------------------------------------------------------------------------------------------

@app.route('/')
def index():
    # Renders the index.html page (home page)
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    success = False
    error_message = None  # Variable to hold the error message

    if request.method == 'POST':
        student_id = request.form['student_id']
        name = request.form['name']
        branch = request.form['branch']
        section = request.form['section']
        room_no = request.form['room_no']
        mobile = request.form['mobile']
        email = request.form['email']

        if 'images' not in request.files:
            return "No images part in the form!"

        images = request.files.getlist('images')  # Get the list of uploaded images

        if not images:
            return "No files uploaded!"

        branch = request.form['branch'].lower()
        section = request.form['section'].lower()
        student_id = request.form['student_id'].lower()

        # Create folders if they don't exist
        datasets_folder = os.path.join('datasets', f"{branch.upper()}")
        branch_folder = os.path.join(datasets_folder, f"{branch}_images")
        student_folder = os.path.join(branch_folder, student_id)
        os.makedirs(student_folder, exist_ok=True)

        # Save images in the student-specific folder
        for image in images:
            if image and image.filename:  # Check for valid files
                image_path = os.path.join(student_folder, secure_filename(image.filename))
                image.save(image_path)
                print(f"Saved image to: {image_path}")

        # Insert data into the correct branch table
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()

            # Check if the student already exists
            cursor.execute(f"SELECT * FROM {branch} WHERE student_id = ?", (student_id,))
            existing_student = cursor.fetchone()

            if existing_student:
                # If student exists, set the error message
                error_message = "Student ID already exists in the database!"
            else:
                # Insert into the specific branch table if the student doesn't exist
                cursor.execute(f'''
                    INSERT INTO {branch} (student_id, name, branch, section, room_no, mobile, email)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (student_id, name, branch, section, room_no, mobile, email))
                conn.commit()  # Ensure data is saved to the database
                success = True

            # Save to Excel
            filename = f'{branch}_{section}.xlsx'
            if not os.path.exists(filename):
                df = pd.DataFrame(columns=['Student ID', 'Name', 'Branch', 'Section', 'Room Number', 'Mobile', 'Email'])
            else:
                df = pd.read_excel(filename)

            # Append or update data in the DataFrame
            new_data = pd.DataFrame([{
                'Student ID': student_id,
                'Name': name,
                'Branch': branch,
                'Section': section,
                'Room Number': room_no,
                'Mobile': mobile,
                'Email': email
            }])

            # Use pd.concat to append the new data to the existing DataFrame
            df = pd.concat([df, new_data], ignore_index=True)

            # Save the updated DataFrame to Excel
            df.to_excel(filename, index=False, engine='openpyxl')

            conn.close()

        except sqlite3.Error as e:
            error_message = f"Error: {e}"
            print(f"Error: {e}")

    return render_template('registration/register.html', success=success, error_message=error_message)

@app.route('/admin')
def admin():
    # Renders the admin.html page (admin panel)
    return render_template('admin/admin.html')

# Default faculty credentials (to be provided by admin)
FACULTY_CREDENTIALS = {
    "faculty123": "password123",
    "faculty456": "securepass456"
}
@app.route('/faculty_login', methods=['GET', 'POST'])
def faculty_login():
    if request.method == 'POST':
        faculty_id = request.form.get('faculty_id')
        password = request.form.get('password')

        # Verify credentials
        if faculty_id in FACULTY_CREDENTIALS and FACULTY_CREDENTIALS[faculty_id] == password:
            session['faculty_id'] = faculty_id  # Store faculty ID in session
            return redirect(url_for('select_class'))  # Redirect to attendance page
        else:
            error_message = "Invalid Faculty ID or Password"
            return render_template('faculty/faculty_login.html', error_message=error_message)
    return render_template('faculty/faculty_login.html')

@app.route('/select_class', methods=['GET', 'POST'])
def select_class():
    if 'faculty_id' not in session:  # Check if faculty is logged in
        return redirect(url_for('faculty/faculty_login'))  # Redirect to login if not authenticated

    if request.method == 'POST':
        class_name = request.form.get('class_name')
        room_number = request.form.get('room_number')
        attendance_data = [
            {'student_id': '21A51A4201', 'name': 'John Doe', 'status': 'Present', 'generated_at': '21-12-2024 & 15:33'},
            {'student_id': '21A51A4202', 'name': 'Jane Smith', 'status': 'Absent', 'generated_at': '21-12-2024 & 15:33'},
            {'student_id': '21A51A4203', 'name': 'Emily Johnson', 'status': 'Present', 'generated_at': '21-12-2024 & 15:33'},
            {'student_id': '21A51A4204', 'name': 'Michael Brown', 'status': 'Present', 'generated_at': '21-12-2024 & 15:33'},
            {'student_id': '21A51A4205', 'name': 'Sarah Davis', 'status': 'Absent', 'generated_at': '21-12-2024 & 15:33'},
            {'student_id': '21A51A4206', 'name': 'Yogeswara Rao', 'status': 'Present', 'generated_at': '21-12-2024 & 15:33'},
        ]
        return render_template('faculty/get_attendance.html', class_name=class_name, room_number=room_number, attendance_data=attendance_data)
    return render_template('faculty/select_class.html')

@app.route('/logout')
def logout():
    session.pop('faculty_id', None)  # Clear the session
    return redirect(url_for('faculty/faculty_login'))

@app.route('/student_attendance', methods=['GET', 'POST'])
def student_attendance():
    roll_number = None
    attendance_data = None
    error_message = None
    if request.method == 'POST':
        roll_number = request.form.get('roll_number')

        #connect to student_attendance.db
        conn = sqlite3.connect('student_attendance.db')
        cursor = conn.cursor()

        try:
            # Query the specific table based on student roll number
            cursor.execute(f'SELECT status, time FROM "{roll_number}"')
            attendance_data = cursor.fetchall()

            if not attendance_data:
                error_message = "No attendance records found for this roll number."

        except sqlite3.Error as e:
            error_message = f"Error occurred: {e}"

        finally:
            conn.close()
    return render_template('student_details/student_attendance.html', roll_number=roll_number, attendance_data=attendance_data, error_message=error_message)

'''
@app.route('/get_attendance', methods=['POST'])
def get_attendance():
    class_name = request.form['class_name']
    room_number = request.form['room_number']
    return render_template('faculty/get_attendance.html', class_name=class_name, room_number=room_number)
'''
'''
@app.route('/faculty')
def faculty():
    # Renders the faculty dashboard page (you can create this page as needed)
    return render_template('faculty.html')

@app.route('/student')
def student():
    # Renders the student page (you can create this page as needed)
    return render_template('student.html')
'''

if __name__ == '__main__':
    app.run(debug=True)
