from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model_generation_time = None

@app.route('/')
def index():
    # Renders the index.html page (home page)
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('registration/register.html')

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
            {'student_id': '21A51A4201', 'name': 'John Doe', 'status': 'Present', 'remarks': 'Good', 'generated_at': '21-12-2024 & 15:33'},
            {'student_id': '21A51A4202', 'name': 'Jane Smith', 'status': 'Absent', 'remarks': 'Good', 'generated_at': '21-12-2024 & 15:33'},
            {'student_id': '21A51A4203', 'name': 'Emily Johnson', 'status': 'Present', 'remarks': 'Good', 'generated_at': '21-12-2024 & 15:33'},
            {'student_id': '21A51A4204', 'name': 'Michael Brown', 'status': 'Present', 'remarks': 'Good', 'generated_at': '21-12-2024 & 15:33'},
            {'student_id': '21A51A4205', 'name': 'Sarah Davis', 'status': 'Absent', 'remarks': 'Good', 'generated_at': '21-12-2024 & 15:33'},
            {'student_id': '21A51A4206', 'name': 'Yogeswara Rao', 'status': 'Present', 'remarks': 'Good', 'generated_at': '21-12-2024 & 15:33'},
        ]
        return render_template('user/get_attendance.html', class_name=class_name, room_number=room_number, attendance_data=attendance_data)
    return render_template('user/select_class.html')

@app.route('/student_attendance', methods=['POST', 'GET'])
def student_attendance():
    roll_number = request.form['roll_number']
    # Simulate fetching data from the database
    attendance_db = {
        '1234': [
            {'date': '2024-12-18', 'status': 'Present'},
            {'date': '2024-12-19', 'status': 'Absent'},
            {'date': '2024-12-20', 'status': 'Present'}
        ],
        '5678': [
            {'date': '2024-12-18', 'status': 'Absent'},
            {'date': '2024-12-19', 'status': 'Present'},
            {'date': '2024-12-20', 'status': 'Present'}
        ]
    }
    attendance_details = attendance_db.get(roll_number, [])
    return render_template('student_attendance.html', roll_number=roll_number, attendance_details=attendance_details)

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
