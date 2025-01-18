import sqlite3

# Connect to SQLite database (creates the file if it doesn't exist)
conn = sqlite3.connect('attendance.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# List of branches for which tables will be created
branch = 'CSM'

# Create a table
cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {branch} (
        student_id TEXT PRIMARY KEY,
        name TEXT,
        branch TEXT,
        section TEXT,
        room_no TEXT,
        mobile number,
        email TEXT
    )
''')

# Commit changes and close the connection
conn.commit()
conn.close()

# Student-specific attendance database
conn_students = sqlite3.connect('student_attendance.db')
cursor_students = conn_students.cursor()

# Example: Create a table for a specific student
student_ids = ['21A51A4201', '21A51A4202', '21A51A4203', '21A51A4204', '21A51A4205', '21A51A4206', '21A51A4207']
for student_id in student_ids:
    cursor_students.execute(f'''
        CREATE TABLE IF NOT EXISTS "{student_id}" (
            status TEXT,
            time DATETIME
        )
    ''')
cursor_students.execute(f''' INSERT INTO "21A51A4201" VALUES ("ABSENT", "09:15") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4201" VALUES ("PRESENT", "10:20") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4201" VALUES ("PRESENT", "11:10") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4201" VALUES ("ABSENT", "12:00") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4201" VALUES ("PRESENT", "13:40") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4202" VALUES ("PRESENT", "09:15") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4202" VALUES ("ABSENT", "10:20") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4202" VALUES ("PRESENT", "11:10") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4202" VALUES ("PRESENT", "12:00") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4202" VALUES ("PRESENT", "13:40") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4203" VALUES ("PRESENT", "09:15") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4203" VALUES ("ABSENT", "10:20") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4203" VALUES ("ABSENT", "11:10") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4203" VALUES ("PRESENT", "12:00") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4203" VALUES ("ABSENT", "13:40") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4204" VALUES ("ABSENT", "09:15") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4204" VALUES ("ABSENT", "10:20") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4204" VALUES ("PRESENT", "11:10") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4204" VALUES ("PRESENT", "12:00") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4204" VALUES ("PRESENT", "13:40") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4205" VALUES ("PRESENT", "09:15") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4205" VALUES ("ABSENT", "10:20") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4205" VALUES ("PRESENT", "11:10") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4205" VALUES ("ABSENT", "12:00") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4205" VALUES ("ABSENT", "13:40") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4206" VALUES ("PRESENT", "09:15") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4206" VALUES ("PRESENT", "10:20") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4206" VALUES ("PRESENT", "11:10") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4206" VALUES ("PRESENT", "12:00") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4206" VALUES ("ABSENT", "13:40") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4207" VALUES ("PRESENT", "09:15") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4207" VALUES ("PRESENT", "10:20") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4207" VALUES ("ABSENT", "11:10") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4207" VALUES ("PRESENT", "12:00") ''')
cursor_students.execute(f''' INSERT INTO "21A51A4207" VALUES ("ABSENT", "13:40") ''')


conn_students.commit()
conn_students.close()

print("Database and tables for each branch created successfully!")
