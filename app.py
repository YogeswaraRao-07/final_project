from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    # Renders the index.html page (home page)
    return render_template('index.html')

@app.route('/admin')
def admin():
    # Renders the admin.html page (admin panel)
    return render_template('/final_project/admin/templates/admin.html')

def select_class():
    return render_template('/final_project/user/templates/select_class.html')

@app.route('/get_attendance', methods=['POST'])
def get_attendance():
    class_name = request.form['class_name']
    room_number = request.form['room_number']

@app.route('/user')
def user():
    # Renders the user dashboard page (you can create this page as needed)
    return render_template('user.html')

@app.route('/student')
def student():
    # Renders the student page (you can create this page as needed)
    return render_template('student.html')

if __name__ == "__main__":
    app.run(debug=True)
