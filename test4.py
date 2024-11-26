import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import face_recognition
import sqlite3
from datetime import datetime
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
import csv


class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance System")
        self.root.geometry("1200x700")

        # Initialize variables
        self.is_camera_running = False
        self.frame = None
        self.yolo_boxes = []
        self.students = []
        self.known_face_encoding = []
        self.known_face_names = []

        # Email configuration
        self.SENDER_EMAIL = "anujladkat8@gmail.com"
        self.SENDER_PASSWORD = "vgrtpagsshoawpch"
        self.RECEIVER_EMAIL = "anujladkat9@gmail.com"

        # Load YOLO model
        self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Initialize database
        self.init_database()

        # Create GUI elements
        self.create_gui()

        # Load known faces
        self.load_known_faces()

    def init_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect('attendance.db')
        self.cursor = self.conn.cursor()

        # Create tables if they don't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                registration_date TEXT
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                date TEXT,
                time TEXT,
                FOREIGN KEY (student_id) REFERENCES students(id)
            )
        ''')

        self.conn.commit()

    def create_gui(self):
        """Create the GUI elements"""
        # Create main frames
        self.left_frame = ttk.Frame(self.root, padding="10")
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.right_frame = ttk.Frame(self.root, padding="10")
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # Configure grid weights
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Camera view
        self.camera_label = ttk.Label(self.left_frame)
        self.camera_label.grid(row=0, column=0, padx=5, pady=5)

        # Control buttons
        self.start_button = ttk.Button(self.left_frame, text="Start Camera", command=self.toggle_camera)
        self.start_button.grid(row=1, column=0, pady=5)

        self.export_button = ttk.Button(self.left_frame, text="Export Attendance", command=self.export_attendance)
        self.export_button.grid(row=2, column=0, pady=5)

        # Attendance display
        self.attendance_tree = ttk.Treeview(self.right_frame, columns=("Name", "Time", "Date"), show="headings")
        self.attendance_tree.heading("Name", text="Name")
        self.attendance_tree.heading("Time", text="Time")
        self.attendance_tree.heading("Date", text="Date")
        self.attendance_tree.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(self.right_frame, orient="vertical", command=self.attendance_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)

        # Configure right frame grid weights
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(0, weight=1)

    def load_known_faces(self):
        """Load known face encodings and names"""
        self.known_face_names = ["Elon Musk", "Varun Kolte", "Shah Rukh", "Reddy Anna",
                                 "Anuj Ladkat", "Yuzra Khan", "Heer Parekh"]
        self.students = self.known_face_names.copy()

        for name in self.known_face_names:
            try:
                img_path = f"Images/{name.replace(' ', '_')}.png"
                image = face_recognition.load_image_file(img_path)
                encoding = face_recognition.face_encodings(image)[0]
                self.known_face_encoding.append(encoding)

                # Add student to database if not exists
                self.cursor.execute("INSERT OR IGNORE INTO students (name, registration_date) VALUES (?, ?)",
                                    (name, datetime.now().strftime("%Y-%m-%d")))
                self.conn.commit()
            except Exception as e:
                print(f"Error loading face for {name}: {e}")

    def detect_faces_yolo(self, frame):
        """Detect faces using YOLO"""
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences = [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.6 and class_id == 0:
                    center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = center_x - w // 2, center_y - h // 2

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.6, nms_threshold=0.4)
        return [boxes[i.flatten()[0]] for i in indices] if len(indices) > 0 else []

    def recognize_faces(self, frame):
        """Recognize faces and update attendance"""
        try:
            face_encodings = face_recognition.face_encodings(frame)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encoding, face_encoding)
                if True in matches:
                    face_distance = face_recognition.face_distance(self.known_face_encoding, face_encoding)
                    best_match_index = np.argmin(face_distance)
                    name = self.known_face_names[best_match_index]

                    if name in self.students:
                        self.students.remove(name)
                        current_time = datetime.now().strftime("%H:%M:%S")
                        current_date = datetime.now().strftime("%Y-%m-%d")

                        # Get student ID
                        self.cursor.execute("SELECT id FROM students WHERE name=?", (name,))
                        student_id = self.cursor.fetchone()[0]

                        # Record attendance
                        self.cursor.execute("""
                            INSERT INTO attendance (student_id, date, time)
                            VALUES (?, ?, ?)
                        """, (student_id, current_date, current_time))
                        self.conn.commit()

                        # Update GUI
                        self.attendance_tree.insert("", 0, values=(name, current_time, current_date))
                        print(f"Attendance logged for: {name} at {current_time}")
        except Exception as e:
            print(f"Error during face recognition: {e}")

    def update_camera(self):
        """Update camera feed"""
        if self.is_camera_running:
            ret, frame = self.video_capture.read()
            if ret:
                self.frame = frame
                # Detect and draw faces
                self.yolo_boxes = self.detect_faces_yolo(frame)
                for (x, y, w, h) in self.yolo_boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Recognize faces
                self.recognize_faces(frame)

                # Convert frame for GUI display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                photo = tk.PhotoImage(data=cv2.imencode('.ppm', frame)[1].tobytes())
                self.camera_label.configure(image=photo)
                self.camera_label.image = photo

            self.root.after(10, self.update_camera)

    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.is_camera_running:
            self.video_capture = cv2.VideoCapture(0)
            self.is_camera_running = True
            self.start_button.configure(text="Stop Camera")
            self.update_camera()
        else:
            self.is_camera_running = False
            self.video_capture.release()
            self.start_button.configure(text="Start Camera")
            self.camera_label.configure(image="")

    def export_attendance(self):
        """Export attendance to CSV and send email"""
        try:
            current_date = datetime.now().strftime("%d-%m-%Y")
            csv_file_path = f"attendance_{current_date}.csv"

            # Export to CSV
            with open(csv_file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Time", "Date"])

                self.cursor.execute("""
                    SELECT students.name, attendance.time, attendance.date 
                    FROM attendance 
                    JOIN students ON attendance.student_id = students.id 
                    WHERE attendance.date = ?
                """, (datetime.now().strftime("%Y-%m-%d"),))

                for row in self.cursor.fetchall():
                    writer.writerow(row)

            # Send email
            self.send_email(csv_file_path)
            messagebox.showinfo("Success", "Attendance exported and email sent successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting attendance: {e}")

    def send_email(self, file_path):
        """Send email with attendance report"""
        message = MIMEMultipart()
        message['From'] = self.SENDER_EMAIL
        message['To'] = self.RECEIVER_EMAIL
        message['Subject'] = f'Attendance Report - {datetime.now().strftime("%d-%m-%Y")}'

        body = f"Please find attached the attendance report for {datetime.now().strftime('%d-%m-%Y')}"
        message.attach(MIMEText(body, 'plain'))

        with open(file_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
        message.attach(part)

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.SENDER_EMAIL, self.SENDER_PASSWORD)
            server.sendmail(self.SENDER_EMAIL, self.RECEIVER_EMAIL, message.as_string())
            server.quit()
            print("Email sent successfully!")
        except Exception as e:
            raise Exception(f"Error sending email: {e}")

    def __del__(self):
        """Cleanup on exit"""
        if hasattr(self, 'conn'):
            self.conn.close()
        if hasattr(self, 'video_capture') and self.video_capture.isOpened():
            self.video_capture.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop()