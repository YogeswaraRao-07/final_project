from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

model_generation_time = None

@app.route('/')
def index():
    # Renders the index.html page (home page)
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('admin/register.html')

@app.route('/admin')
def admin():
    # Renders the admin.html page (admin panel)
    return render_template('admin/admin.html')

@app.route('/select_class', methods=['GET', 'POST'])
def select_class():
    if request.method == 'POST':
        # Get form data
        class_name = request.form.get('class_name')  # Safe access
        room_number = request.form.get('room_number')  # Safe access

        # Render get_attendance.html with the received data
        return render_template('user/get_attendance.html', class_name=class_name, room_number=room_number)

    # For GET request, render the select_class.html page
    return render_template('user/select_class.html')

@app.route('/get_attendance', methods=['POST'])
def get_attendance():
    class_name = request.form['class_name']
    room_number = request.form['room_number']
    return render_template('user/get_attendance.html', class_name=class_name, room_number=room_number)

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

if __name__ == "__main__":
    app.run(debug=True)
