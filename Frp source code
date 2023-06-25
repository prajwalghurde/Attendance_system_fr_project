#import all necessary packages from cmd
import tkinter as tk
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
from PIL import ImageTk, Image


# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load known face encodings and names

prajwal_image = face_recognition.load_image_file("prajwal.jpg")
prajwal_encoding = face_recognition.face_encodings(prajwal_image)[0]


# Create a dictionary of known face encodings and names
#Add multiple known faces seperating them with comma
known_faces = {
    
    tuple(prajwal_encoding): "prajwal"
}

# Initialize attendance variables
students = set(known_faces.values())
attendance = set()

# Get current date
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


def mark_attendance(name):
    # Open CSV file and mark attendance
    with open(current_date + '.csv', 'a', newline='') as f:
        lnwriter = csv.writer(f)
        current_time = now.strftime("%H:%M:%S")
        lnwriter.writerow([name, current_time])
        attendance.add(name)
    update_attendance_status()  # Update the attendance status immediately after marking attendance


def detect_faces():
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Face recognition code
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(list(known_faces.keys()), face_encoding, tolerance=0.5)
        name = "UNKNOWN"

        if True in matches:
            match_index = np.argmax(matches)
            name = list(known_faces.values())[match_index]

        face_names.append(name)

        if name in students and name not in attendance:
            mark_attendance(name)

    # Display the frame with face names
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Update the GUI image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (800, 600))
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Schedule the next face detection
    video_label.after(1, detect_faces)


def button_click():
    print("Present marked!")


# Create the GUI window
window = tk.Tk()
window.title("Attendance System")

# Create a label to display the video stream
video_label = tk.Label(window)
video_label.pack()

# Create a button
button = tk.Button(window, text="CLICK HERE!", command=button_click, bg="blue", fg="white", font=("Arial", 16))
button.pack(pady=10)

# Create a label for attendance status
attendance_label = tk.Label(window, text="Attendance Status:", font=("Arial", 14))
attendance_label.pack()

# Create a label to display the current attendance count
count_label = tk.Label(window, text="Total Attendance: 0", font=("Arial", 12))
count_label.pack()

# Create a listbox to display the names of students present
attendance_listbox = tk.Listbox(window, font=("Arial", 12), width=30, height=10)
attendance_listbox.pack()

def update_attendance_status():
    count = len(attendance)
    count_label.config(text="Total Attendance: " + str(count))
    attendance_listbox.delete(0, tk.END)
    for student in attendance:
        attendance_listbox.insert(tk.END, student)

# Start face detection
detect_faces()
update_attendance_status()

# Start the GUI event loop
window.mainloop()

# Release video capture
video_capture.release()
