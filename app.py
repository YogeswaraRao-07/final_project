import openpyxl
import os
import sqlite3
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
# import shutil
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from tensorflow.keras.applications import mobilenet_v2
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# import matplotlib.pyplot as plt
# import numpy as np
# from threading import Thread

app = Flask(__name__)

# app.secret_key = 'your_secret_key'  # Add a secret key for flash messages
# app.config['DATASET_UPLOAD_FOLDER'] = r'C:/Users/yogesh/Downloads/final_project-main/final_project-main/final_project-main/datasets/csm/'
# app.config['IMAGE_UPLOAD_FOLDER'] = r'C:/Users/yogesh/Downloads/final_project-main/final_project-main/final_project-main/images'
# app.config['TEST_IMAGES_DIR'] = r'C:/Users/yogesh/Downloads/final_project-main/final_project-main/final_project-main/datasets/csmt/f'
# app.config['MODEL_PATH'] = r'C:/Users/yogesh/Downloads/final_project-main/final_project-main/final_project-main/models/my_model.h5'
# #labels
# train_dir = r"C:/Users/yogesh/Downloads/final_project-main/final_project-main/final_project-main/datasets/csmp/train"
# IMG_HEIGHT, IMG_WIDTH = 128, 128
# BATCH_SIZE = 16
# # Create generators
# train_gen = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(
#     train_dir,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='categorical'
# )
# # Load the model
# model = load_model(app.config['MODEL_PATH'])
#
# IMG_HEIGHT = 224  # Example height
# IMG_WIDTH = 224   # Example width
# class_labels = ['class1', 'class2', 'class3']  # Example class labels

model_generation_time = None

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

@app.route('/select_class', methods=['GET', 'POST'])
def select_class():
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
        return render_template('user/get_attendance.html', class_name=class_name, room_number=room_number, attendance_data=attendance_data)
    return render_template('user/select_class.html')

@app.route('/student_attendance', methods=['GET', 'POST'])
def student_attendance():
    student_data = None
    error_message = None
    if request.method == 'POST':
        roll_number = request.form.get('roll_number')
        attendance_data = {
            "1001": {
                "name": "John Doe",
                "attendance": "85%",
                "details": [
                    {"date": "2024-12-01", "subject": "Mathematics", "status": "Present"},
                    {"date": "2024-12-02", "subject": "Physics", "status": "Absent"},
                    {"date": "2024-12-03", "subject": "Chemistry", "status": "Present"},
                ],
            },
            "1002": {
                "name": "Jane Smith",
                "attendance": "90%",
                "details": [
                    {"date": "2024-12-01", "subject": "Mathematics", "status": "Present"},
                    {"date": "2024-12-02", "subject": "Physics", "status": "Present"},
                    {"date": "2024-12-03", "subject": "Chemistry", "status": "Present"},
                ],
            },
        }
        student_data = attendance_data.get(roll_number)
        if not student_data:
            error_message = "No record found for the entered roll number."
    return render_template('student_details/student_attendance.html', student_data=student_data, error_message=error_message)

'''
@app.route('/get_attendance', methods=['POST'])
def get_attendance():
    class_name = request.form['class_name']
    room_number = request.form['room_number']
    return render_template('user/get_attendance.html', class_name=class_name, room_number=room_number)
'''
'''
@app.route('/user')
def user():
    # Renders the user dashboard page (you can create this page as needed)
    return render_template('user.html')

@app.route('/student')
def student():
    # Renders the student page (you can create this page as needed)
    return render_template('student.html')
'''

if __name__ == '__main__':
    app.run(debug=True)
