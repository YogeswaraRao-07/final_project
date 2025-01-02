import sqlite3

# Connect to SQLite database (creates the file if it doesn't exist)
conn = sqlite3.connect('attendance.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# List of branches for which tables will be created
branches = ['CSM', 'CSE', 'CSD', 'ECE', 'EEE']

# Create a table for each branch
for branch in branches:
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

cursor.execute(f''' CREATE TABLE IF NOT EXISTS registration (branch TEXT, model_file LONGBLOB) ''')

# Commit changes and close the connection
conn.commit()
conn.close()

print("Database and tables for each branch created successfully!")
