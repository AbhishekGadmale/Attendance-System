import cv2
import os
from flask import Flask, render_template, redirect, url_for, request
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from flask import send_file

# Defining Flask App
app = Flask(__name__)

# Number of images to take for each user
nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")

# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Path to the single attendance file
attendance_file = 'Attendance/Attendance.csv'

# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# If the single attendance file doesn't exist, create it with headers
if not os.path.isfile(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time,Date\n')  # Add headers


# Get the number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# Extract the face from an image
def extract_faces(img):
    # Check if img is not None and has a valid size
    if img is not None and img.size != 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    else:
        return []



# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# Train the model on all the faces available in the faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from the single attendance file
def extract_attendance():
    df = pd.read_csv(attendance_file)
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    dates = df['Date']
    classes = df['Class']  # Add class information
    l = len(df)
    return names, rolls, times, dates, classes, l


# Add attendance of a specific user
def add_attendance(name, selected_class):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    current_date = date.today().strftime("%Y-%m-%d")

    # Convert userid safely
    try:
        userid = int(userid)
    except ValueError:
        return "Invalid user ID format", 400  # Return an error if conversion fails

    # Check if the user is already marked for the current date and class
    df = pd.read_csv(attendance_file)

    if not ((df['Roll'] == userid) & (df['Date'] == current_date) & (df['Class'] == selected_class)).any():
        with open(attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{current_date},{selected_class}')


################## ROUTING FUNCTIONS #######################

# Main page
@app.route('/')
def home():
    df = pd.read_csv(attendance_file)

    names = df['Name'].tolist()
    rolls = df['Roll'].tolist()
    times = df['Time'].tolist()
    dates = df['Date'].tolist()
    classes = df['Class'].tolist()
    l = len(df)

    # ✅ Define attendance_counts BEFORE return
    attendance_counts = df['Class'].value_counts().to_dict()

    return render_template(
        'home.html', 
        names=names, rolls=rolls, times=times, dates=dates, classes=classes, 
        l=l, totalreg=totalreg(), attendance_counts=attendance_counts
    )

    # Calculate attendance counts by class
    attendance_counts = df['Class'].value_counts().to_dict()

# Start attendance
# Modify Start Attendance Route to include class selection
@app.route('/start', methods=['GET'])
def start():
    selected_class = request.args.get('class')  # Get selected class
    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person, selected_class)  # Pass selected class
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, dates, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, dates=dates, l=l, totalreg=totalreg())
# Add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        
        if j == nimgs * 5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print('Training Model')
    train_model()
    
    # ✅ Extract all required values before returning the template
    names, rolls, times, dates, classes, l = extract_attendance()
    attendance_counts = pd.read_csv(attendance_file)['Class'].value_counts().to_dict()  # ✅ Fix: Define attendance_counts

    return render_template('home.html', 
                           names=names, rolls=rolls, times=times, dates=dates, 
                           classes=classes, l=l, totalreg=totalreg(), attendance_counts=attendance_counts)

def get_user_info(user_roll):
    df = pd.read_csv(attendance_file)
    user_data = df[df['Roll'] == int(user_roll)]
    return user_data

# Route to view individual record
@app.route('/record/<int:roll>', methods=['GET'])
def view_record(roll):
    user_data = get_user_info(roll)
    return render_template('view_record.html', user_data=user_data)

# Route to delete user
@app.route('/delete/<int:roll>', methods=['POST'])
def delete_user(roll):
    # Remove images
    user_folder = f"static/faces/{roll}"
    if os.path.isdir(user_folder):
        for img in os.listdir(user_folder):
            os.remove(f"{user_folder}/{img}")
        os.rmdir(user_folder)
    
    # Remove from attendance
    df = pd.read_csv(attendance_file)
    df = df[df['Roll'] != roll]
    df.to_csv(attendance_file, index=False)
    
    # Re-train the model after deletion
    train_model()
    
    return redirect(url_for('home'))
@app.route('/filter', methods=['GET'])
def filter_attendance():
    selected_class = request.args.get('class')
    df = pd.read_csv(attendance_file)
    filtered_df = df[df['Class'] == selected_class]
    names = filtered_df['Name']
    rolls = filtered_df['Roll']
    times = filtered_df['Time']
    dates = filtered_df['Date']
    classes = filtered_df['Class']
    l = len(filtered_df)

    # Fix: Add attendance_counts
    attendance_counts = df['Class'].value_counts().to_dict()

    return render_template(
        'home.html', names=names, rolls=rolls, times=times, 
        dates=dates, classes=classes, l=l, 
        totalreg=totalreg(), attendance_counts=attendance_counts
    )


@app.route('/clear', methods=['POST'])
def clear_attendance():
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time,Date,Class\n')  # Reset to headers only
    return redirect(url_for('home'))

@app.route('/export_class', methods=['GET'])
def export_class_attendance():
    selected_class = request.args.get('class')
    df = pd.read_csv(attendance_file)
    filtered_df = df[df['Class'] == selected_class]
    export_path = f'Attendance/Attendance_{selected_class}.csv'
    filtered_df.to_csv(export_path, index=False)
    return send_file(export_path, as_attachment=True)



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
