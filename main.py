import cv2
import dlib
import numpy as np
import pygame
import threading
import time
import tkinter as tk
from tkinter import ttk, Frame, Label, Button, Scale, HORIZONTAL, Entry, StringVar, messagebox
from PIL import Image, ImageTk
from scipy.spatial import distance
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import datetime
from twilio.rest import Client

# Initialize pygame for alert sound
pygame.mixer.init()

class DrowsinessDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Drowsiness Detection System")
        self.root.geometry("1200x750")
        self.root.configure(bg="#f0f2f5")
        self.root.resizable(True, True)
        
        # Initialize variables
        self.is_running = False
        self.frame = None
        self.eye_aspect_ratio_threshold = 0.25
        self.eye_aspect_ratio_consecutive_frames = 20
        self.counter = 0
        self.alarm_on = False
        
        # Yawn detection variables
        self.yawn_threshold = 30  # Distance threshold for yawn detection
        self.yawn_counter = 0
        self.yawn_consecutive_frames = 15
        self.yawn_alarm_on = False
        
        # Emergency contact variables
        self.drowsy_start_time = None
        self.emergency_timeout = 15  # seconds
        self.emergency_triggered = False
        self.emergency_contact_name = StringVar()
        self.emergency_contact_phone = StringVar()
        self.emergency_contact_email = StringVar()
        
        # Statistics variables
        self.drowsy_episodes = 0
        self.yawn_episodes = 0
        self.last_alert_time = None
        self.total_monitoring_time = 0
        self.monitoring_start_time = None
        
        # Load face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        # You need to download shape_predictor_68_face_landmarks.dat from:
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Define eye landmarks indices
        self.left_eye_start = 42
        self.left_eye_end = 48
        self.right_eye_start = 36
        self.right_eye_end = 42
        
        # Define mouth landmarks indices
        self.mouth_start = 48
        self.mouth_end = 68
        
        # Load alarm sounds
        pygame.mixer.music.load("alarm.wav")  # Create an alarm.wav file or use any sound file
        
        # Create UI components
        self.create_ui()
        
        # Initialize session data
        self.session_data = {
            "start_time": None,
            "end_time": None,
            "drowsy_episodes": 0,
            "yawn_episodes": 0,
            "emergency_contacts": 0
        }
        
    def create_ui(self):
        # Create main frame containers
        top_frame = Frame(self.root, bg="#f0f2f5")
        top_frame.pack(side="top", fill="x", padx=20, pady=10)
        
        main_frame = Frame(self.root, bg="#f0f2f5")
        main_frame.pack(side="top", fill="both", expand=True, padx=20, pady=10)
        
        # Create title
        title_label = Label(top_frame, text="Enhanced Drowsiness Detection System", font=("Helvetica", 24, "bold"), bg="#f0f2f5", fg="#2e4057")
        title_label.pack(pady=10)
        
        desc_label = Label(top_frame, text="Monitor your alertness level while driving or working", font=("Helvetica", 12), bg="#f0f2f5", fg="#555")
        desc_label.pack(pady=5)
        
        # Create left panel for video feed
        self.video_panel = Frame(main_frame, bg="#ffffff", highlightbackground="#ddd", highlightthickness=1)
        self.video_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        # Video label
        self.video_label = Label(self.video_panel)
        self.video_label.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Create right panel for controls
        control_panel = Frame(main_frame, bg="#ffffff", width=350, highlightbackground="#ddd", highlightthickness=1)
        control_panel.pack(side="right", fill="y", padx=10, pady=10)
        control_panel.pack_propagate(False)
        
        # Create tabs for organization
        tab_control = ttk.Notebook(control_panel)
        
        # Status tab
        status_tab = Frame(tab_control, bg="#ffffff")
        settings_tab = Frame(tab_control, bg="#ffffff")
        emergency_tab = Frame(tab_control, bg="#ffffff")
        stats_tab = Frame(tab_control, bg="#ffffff")
        
        tab_control.add(status_tab, text="Status")
        tab_control.add(settings_tab, text="Settings")
        tab_control.add(emergency_tab, text="Emergency")
        tab_control.add(stats_tab, text="Statistics")
        
        tab_control.pack(expand=1, fill="both", padx=5, pady=5)
        
        # Status section
        status_frame = Frame(status_tab, bg="#ffffff")
        status_frame.pack(fill="x", padx=20, pady=20)
        
        Label(status_frame, text="Status", font=("Helvetica", 16, "bold"), bg="#ffffff", fg="#2e4057").pack(anchor="w")
        
        self.status_label = Label(status_frame, text="Not Monitoring", font=("Helvetica", 14), bg="#ffffff", fg="#f44336")
        self.status_label.pack(anchor="w", pady=5)
        
        self.eye_status_label = Label(status_frame, text="Eye Status: -", font=("Helvetica", 12), bg="#ffffff")
        self.eye_status_label.pack(anchor="w", pady=2)
        
        self.ear_value_label = Label(status_frame, text="EAR: -", font=("Helvetica", 12), bg="#ffffff")
        self.ear_value_label.pack(anchor="w", pady=2)
        
        self.yawn_status_label = Label(status_frame, text="Yawn Status: -", font=("Helvetica", 12), bg="#ffffff")
        self.yawn_status_label.pack(anchor="w", pady=2)
        
        self.mouth_distance_label = Label(status_frame, text="Mouth Distance: -", font=("Helvetica", 12), bg="#ffffff")
        self.mouth_distance_label.pack(anchor="w", pady=2)
        
        self.monitoring_time_label = Label(status_frame, text="Monitoring Time: 00:00:00", font=("Helvetica", 12), bg="#ffffff")
        self.monitoring_time_label.pack(anchor="w", pady=10)
        
        # Sensitivity settings tab
        settings_frame = Frame(settings_tab, bg="#ffffff")
        settings_frame.pack(fill="x", padx=20, pady=20)
        
        Label(settings_frame, text="Sensitivity Settings", font=("Helvetica", 16, "bold"), bg="#ffffff", fg="#2e4057").pack(anchor="w")
        
        # Eye detection settings
        Label(settings_frame, text="Eye Detection", font=("Helvetica", 14), bg="#ffffff", fg="#2e4057").pack(anchor="w", pady=5)
        
        Label(settings_frame, text="EAR Threshold:", font=("Helvetica", 12), bg="#ffffff").pack(anchor="w", pady=2)
        
        self.threshold_scale = Scale(settings_frame, from_=0.15, to=0.35, orient=HORIZONTAL, 
                                     resolution=0.01, length=250, bg="#ffffff", highlightthickness=0,
                                     command=self.update_threshold)
        self.threshold_scale.set(self.eye_aspect_ratio_threshold)
        self.threshold_scale.pack(anchor="w")
        
        Label(settings_frame, text="Consecutive Frames:", font=("Helvetica", 12), bg="#ffffff").pack(anchor="w", pady=2)
        
        self.frames_scale = Scale(settings_frame, from_=5, to=50, orient=HORIZONTAL, 
                                  resolution=1, length=250, bg="#ffffff", highlightthickness=0,
                                  command=self.update_frames)
        self.frames_scale.set(self.eye_aspect_ratio_consecutive_frames)
        self.frames_scale.pack(anchor="w")
        
        # Yawn detection settings
        Label(settings_frame, text="Yawn Detection", font=("Helvetica", 14), bg="#ffffff", fg="#2e4057").pack(anchor="w", pady=5)
        
        Label(settings_frame, text="Yawn Threshold:", font=("Helvetica", 12), bg="#ffffff").pack(anchor="w", pady=2)
        
        self.yawn_threshold_scale = Scale(settings_frame, from_=20, to=40, orient=HORIZONTAL, 
                                     resolution=1, length=250, bg="#ffffff", highlightthickness=0,
                                     command=self.update_yawn_threshold)
        self.yawn_threshold_scale.set(self.yawn_threshold)
        self.yawn_threshold_scale.pack(anchor="w")
        
        Label(settings_frame, text="Consecutive Frames:", font=("Helvetica", 12), bg="#ffffff").pack(anchor="w", pady=2)
        
        self.yawn_frames_scale = Scale(settings_frame, from_=5, to=30, orient=HORIZONTAL, 
                                  resolution=1, length=250, bg="#ffffff", highlightthickness=0,
                                  command=self.update_yawn_frames)
        self.yawn_frames_scale.set(self.yawn_consecutive_frames)
        self.yawn_frames_scale.pack(anchor="w")
        
        # Emergency contact tab
        emergency_frame = Frame(emergency_tab, bg="#ffffff")
        emergency_frame.pack(fill="x", padx=20, pady=20)
        
        Label(emergency_frame, text="Emergency Contact", font=("Helvetica", 16, "bold"), bg="#ffffff", fg="#2e4057").pack(anchor="w")
        Label(emergency_frame, text="Will be notified if drowsy for more than 15 seconds", font=("Helvetica", 10), bg="#ffffff", fg="#555").pack(anchor="w", pady=(0, 10))
        
        Label(emergency_frame, text="Contact Name:", font=("Helvetica", 12), bg="#ffffff").pack(anchor="w", pady=2)
        Entry(emergency_frame, textvariable=self.emergency_contact_name, width=30).pack(anchor="w", pady=2)
        
        Label(emergency_frame, text="Contact Phone:", font=("Helvetica", 12), bg="#ffffff").pack(anchor="w", pady=2)
        Entry(emergency_frame, textvariable=self.emergency_contact_phone, width=30).pack(anchor="w", pady=2)
        
        Label(emergency_frame, text="Contact Email:", font=("Helvetica", 12), bg="#ffffff").pack(anchor="w", pady=2)
        Entry(emergency_frame, textvariable=self.emergency_contact_email, width=30).pack(anchor="w", pady=2)
        
        Button(emergency_frame, text="Save Contact", font=("Helvetica", 12), 
              bg="#4caf50", fg="white", width=15,
              command=self.save_emergency_contact).pack(anchor="w", pady=10)
        
        # Test emergency contact button
        Button(emergency_frame, text="Test Contact", font=("Helvetica", 12), 
              bg="#ff9800", fg="white", width=15,
              command=self.test_emergency_contact).pack(anchor="w", pady=5)
        
        # Statistics tab
        stats_frame = Frame(stats_tab, bg="#ffffff")
        stats_frame.pack(fill="x", padx=20, pady=20)
        
        Label(stats_frame, text="Session Statistics", font=("Helvetica", 16, "bold"), bg="#ffffff", fg="#2e4057").pack(anchor="w")
        
        self.drowsy_count_label = Label(stats_frame, text="Drowsy Episodes: 0", font=("Helvetica", 12), bg="#ffffff")
        self.drowsy_count_label.pack(anchor="w", pady=2)
        
        self.yawn_count_label = Label(stats_frame, text="Yawn Episodes: 0", font=("Helvetica", 12), bg="#ffffff")
        self.yawn_count_label.pack(anchor="w", pady=2)
        
        self.alert_sent_label = Label(stats_frame, text="Emergency Alerts Sent: 0", font=("Helvetica", 12), bg="#ffffff")
        self.alert_sent_label.pack(anchor="w", pady=2)
        
        self.last_alert_label = Label(stats_frame, text="Last Alert: -", font=("Helvetica", 12), bg="#ffffff")
        self.last_alert_label.pack(anchor="w", pady=2)
        
        Button(stats_frame, text="Export Statistics", font=("Helvetica", 12), 
              bg="#2196f3", fg="white", width=15,
              command=self.export_statistics).pack(anchor="w", pady=10)
        
        # Control buttons at the bottom of the control panel
        buttons_frame = Frame(control_panel, bg="#ffffff")
        buttons_frame.pack(fill="x", padx=20, pady=20, side="bottom")
        
        self.start_button = Button(buttons_frame, text="Start Monitoring", font=("Helvetica", 12), 
                                  bg="#4caf50", fg="white", width=15, height=2,
                                  command=self.toggle_monitoring)
        self.start_button.pack(side="left", padx=5)
        
        self.stop_button = Button(buttons_frame, text="Exit", font=("Helvetica", 12), 
                                 bg="#f44336", fg="white", width=15, height=2,
                                 command=self.exit_application)
        self.stop_button.pack(side="right", padx=5)
    
    def update_threshold(self, val):
        self.eye_aspect_ratio_threshold = float(val)
    
    def update_frames(self, val):
        self.eye_aspect_ratio_consecutive_frames = int(val)
    
    def update_yawn_threshold(self, val):
        self.yawn_threshold = int(val)
    
    def update_yawn_frames(self, val):
        self.yawn_consecutive_frames = int(val)
    
    def save_emergency_contact(self):
        if not self.emergency_contact_name.get() or not (self.emergency_contact_phone.get() or self.emergency_contact_email.get()):
            messagebox.showwarning("Missing Information", "Please provide at least a name and either a phone number or email.")
            return
        
        messagebox.showinfo("Success", "Emergency contact saved successfully!")
    
    def test_emergency_contact(self):
        if not self.emergency_contact_name.get() or not (self.emergency_contact_phone.get() or self.emergency_contact_email.get()):
            messagebox.showwarning("Missing Information", "Please provide at least a name and either a phone number or email.")
            return
        
        try:
            self.send_emergency_alert(test=True)
            messagebox.showinfo("Success", "Test alert sent successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send test alert: {str(e)}")
    
    def toggle_monitoring(self):
        if not self.is_running:
            self.is_running = True
            self.start_button.config(text="Stop Monitoring", bg="#ff9800")
            self.status_label.config(text="Monitoring Active", fg="#4caf50")
            
            # Reset statistics for new session
            self.drowsy_episodes = 0
            self.yawn_episodes = 0
            self.session_data["emergency_contacts"] = 0
            self.monitoring_start_time = time.time()
            self.session_data["start_time"] = datetime.datetime.now()
            
            # Start timer update
            self.update_monitoring_time()
            
            # Start video stream in a separate thread
            threading.Thread(target=self.start_video_stream, daemon=True).start()
        else:
            self.is_running = False
            self.start_button.config(text="Start Monitoring", bg="#4caf50")
            self.status_label.config(text="Not Monitoring", fg="#f44336")
            self.eye_status_label.config(text="Eye Status: -")
            self.ear_value_label.config(text="EAR: -")
            self.yawn_status_label.config(text="Yawn Status: -")
            self.mouth_distance_label.config(text="Mouth Distance: -")
            
            # Save session end time
            self.session_data["end_time"] = datetime.datetime.now()
            
            if self.alarm_on:
                self.stop_alarm()
            if self.yawn_alarm_on:
                self.stop_yawn_alarm()
    
    def update_monitoring_time(self):
        if self.is_running and self.monitoring_start_time:
            elapsed = int(time.time() - self.monitoring_start_time)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.monitoring_time_label.config(text=f"Monitoring Time: {time_str}")
            
            # Schedule next update
            self.root.after(1000, self.update_monitoring_time)
    
    def start_video_stream(self):
        cap = cv2.VideoCapture(0)  # Use 0 for default camera
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray, 0)
            
            # Process each face
            for face in faces:
                # Get facial landmarks
                landmarks = self.predictor(gray, face)
                
                # Calculate eye aspect ratio
                left_eye = []
                right_eye = []
                
                for n in range(self.left_eye_start, self.left_eye_end):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    left_eye.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                for n in range(self.right_eye_start, self.right_eye_end):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    right_eye.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Calculate EAR for both eyes
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                
                # Average the EAR
                ear = (left_ear + right_ear) / 2.0
                
                # Update UI with current EAR value
                self.root.after(1, lambda e=ear: self.ear_value_label.config(text=f"EAR: {e:.2f}"))
                
                # Draw eye contours
                leftEyeHull = cv2.convexHull(np.array(left_eye))
                rightEyeHull = cv2.convexHull(np.array(right_eye))
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                
                # Get mouth landmarks for yawn detection
                mouth = []
                for n in range(self.mouth_start, self.mouth_end):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    mouth.append((x, y))
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Calculate mouth aspect ratio or vertical distance
                mouth_top = landmarks.part(62).y  # Top of the mouth
                mouth_bottom = landmarks.part(66).y  # Bottom of the mouth
                mouth_distance = abs(mouth_top - mouth_bottom)
                
                # Update UI with current mouth distance
                self.root.after(1, lambda d=mouth_distance: self.mouth_distance_label.config(text=f"Mouth Distance: {d}"))
                
                # Draw mouth contour
                mouthHull = cv2.convexHull(np.array(mouth))
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                
                # Process yawn detection
                if mouth_distance > self.yawn_threshold:
                    self.yawn_counter += 1
                    if self.yawn_counter >= self.yawn_consecutive_frames:
                        # Alert for yawn
                        if not self.yawn_alarm_on:
                            self.yawn_alarm_on = True
                            threading.Thread(target=self.start_yawn_alarm, daemon=True).start()
                            
                            # Increment yawn counter
                            self.yawn_episodes += 1
                            self.session_data["yawn_episodes"] = self.yawn_episodes
                            self.root.after(1, lambda: self.yawn_count_label.config(text=f"Yawn Episodes: {self.yawn_episodes}"))
                        
                        cv2.putText(frame, "YAWN DETECTED!", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.root.after(1, lambda: self.yawn_status_label.config(text="Yawn Status: YAWNING", fg="#f44336"))
                else:
                    self.yawn_counter = 0
                    if self.yawn_alarm_on:
                        self.stop_yawn_alarm()
                    self.root.after(1, lambda: self.yawn_status_label.config(text="Yawn Status: NORMAL", fg="#4caf50"))
                
                # Check if eyes are closed based on EAR
                if ear < self.eye_aspect_ratio_threshold:
                    self.counter += 1
                    
                    if self.counter >= self.eye_aspect_ratio_consecutive_frames:
                        # Start timing for emergency contact
                        if self.drowsy_start_time is None:
                            self.drowsy_start_time = time.time()
                            # Increment drowsy episode counter
                            self.drowsy_episodes += 1
                            self.session_data["drowsy_episodes"] = self.drowsy_episodes
                            self.root.after(1, lambda: self.drowsy_count_label.config(text=f"Drowsy Episodes: {self.drowsy_episodes}"))
                        
                        # Check if drowsy for more than emergency timeout
                        if not self.emergency_triggered and self.drowsy_start_time is not None:
                            drowsy_duration = time.time() - self.drowsy_start_time
                            if drowsy_duration > self.emergency_timeout:
                                self.emergency_triggered = True
                                threading.Thread(target=self.send_emergency_alert, daemon=True).start()
                        
                        # Alert the user
                        if not self.alarm_on:
                            self.alarm_on = True
                            threading.Thread(target=self.start_alarm, daemon=True).start()
                        
                        # Draw alert on frame
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Update UI
                        self.root.after(1, lambda: self.eye_status_label.config(text="Eye Status: CLOSED", fg="#f44336"))
                else:
                    self.counter = 0
                    self.drowsy_start_time = None
                    self.emergency_triggered = False
                    
                    if self.alarm_on:
                        self.stop_alarm()
                    
                    # Update UI
                    self.root.after(1, lambda: self.eye_status_label.config(text="Eye Status: OPEN", fg="#4caf50"))
            
            # Convert frame to display in UI
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_frame()
            
        cap.release()
    
    def eye_aspect_ratio(self, eye):
        # Compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        
        # Compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = distance.euclidean(eye[0], eye[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def display_frame(self):
        if self.frame is not None:
            # Resize frame for display if needed
            h, w = self.frame.shape[:2]
            max_w = 800  # Maximum width for display
            if w > max_w:
                ratio = max_w / w
                new_h = int(h * ratio)
                self.frame = cv2.resize(self.frame, (max_w, new_h))
            
            img = Image.fromarray(self.frame)
            img = ImageTk.PhotoImage(image=img)
            
            self.video_label.config(image=img)
            self.video_label.image = img
    
    def start_alarm(self):
        pygame.mixer.music.play(-1)  # -1 loops the sound
        
        # Change UI elements to alert state
        self.root.after(1, lambda: self.status_label.config(text="DROWSINESS DETECTED!", fg="#f44336"))
        self.last_alert_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.root.after(1, lambda: self.last_alert_label.config(text=f"Last Alert: {self.last_alert_time}"))
    
    def stop_alarm(self):
        pygame.mixer.music.stop()
        self.alarm_on = False
        
        # Reset UI elements to normal state
        self.root.after(1, lambda: self.status_label.config(text="Monitoring Active", fg="#4caf50"))
    
    def start_yawn_alarm(self):
        # We could use a different sound for yawn alert
        # For now, just update the UI
        self.root.after(1, lambda: self.status_label.config(text="YAWN DETECTED!", fg="#ff9800"))
    
    def stop_yawn_alarm(self):
        self.yawn_alarm_on = False
        
        # Reset UI elements to normal state if no other alarms are active
        if not self.alarm_on:
            self.root.after(1, lambda: self.status_label.config(text="Monitoring Active", fg="#4caf50"))
    
    def send_emergency_alert(self, test=False):
        """Send emergency alert to the designated contact via email and/or SMS"""
        # Update statistics
        if not test:
            self.session_data["emergency_contacts"] += 1
            self.root.after(1, lambda: self.alert_sent_label.config(text=f"Emergency Alerts Sent: {self.session_data['emergency_contacts']}"))
        
        name = self.emergency_contact_name.get()
        email = self.emergency_contact_email.get()
        phone = self.emergency_contact_phone.get()
        
        alert_message = f"EMERGENCY ALERT: {name if name else 'Emergency contact'}, the driver/user has been detected as drowsy for an extended period."
        if test:
            alert_message += " This is a TEST alert."
        
        # Send email if provided
        if email:
            try:
                self.send_email(email, "DROWSINESS ALERT - Urgent!", alert_message)
            except:
                print("Failed to send email alert")
        
        # Send SMS if phone number provided
        if phone:
            try:
                self.send_sms(phone, alert_message)
            except:
                print("Failed to send SMS alert")
    
    def send_email(self, recipient, subject, body):
        """Send email using SMTP - For demonstration purposes only"""
        # NOTE: In a real application, you would use proper authentication
        try:
            # This is just a demonstration - in a real app you would use a real SMTP server
            print(f"Simulating sending email to {recipient}")
            print(f"Subject: {subject}")
            print(f"Body: {body}")
        except Exception as e:
            print(f"Email sending error: {e}")

    def send_sms(self, recipient, body):
        """Send SMS using Twilio - For demonstration purposes only"""
        # NOTE: In a real application, you would use proper authentication
        try:
            # This is just a demonstration - in a real app you would use actual Twilio credentials
            print(f"Simulating sending SMS to {recipient}")
            print(f"Message: {body}")
        except Exception as e:
            print(f"SMS sending error: {e}")
    
    def export_statistics(self):
        """Export session statistics to a text file"""
        if not self.session_data["start_time"]:
            messagebox.showinfo("No Data", "No monitoring session data available to export.")
            return
        
        try:
            # Create a reports directory if it doesn't exist
            if not os.path.exists("reports"):
                os.makedirs("reports")
            
            # Create a filename with current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reports/drowsiness_report_{timestamp}.txt"
            
            # Calculate session duration
            end_time = self.session_data["end_time"] or datetime.datetime.now()
            duration = end_time - self.session_data["start_time"]
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Format the duration string
            duration_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            
            with open(filename, "w") as f:
                f.write("===== DROWSINESS DETECTION SYSTEM - SESSION REPORT =====\n\n")
                f.write(f"Session Start: {self.session_data['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Session End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Session Duration: {duration_str}\n\n")
                f.write(f"Drowsy Episodes Detected: {self.session_data['drowsy_episodes']}\n")
                f.write(f"Yawn Episodes Detected: {self.session_data['yawn_episodes']}\n")
                f.write(f"Emergency Alerts Sent: {self.session_data['emergency_contacts']}\n\n")
                f.write("Settings Used:\n")
                f.write(f"- Eye Aspect Ratio Threshold: {self.eye_aspect_ratio_threshold}\n")
                f.write(f"- Consecutive Frames for Drowsiness: {self.eye_aspect_ratio_consecutive_frames}\n")
                f.write(f"- Yawn Threshold: {self.yawn_threshold}\n")
                f.write(f"- Consecutive Frames for Yawn: {self.yawn_consecutive_frames}\n")
                f.write(f"- Emergency Contact Timeout: {self.emergency_timeout} seconds\n\n")
                f.write("===== END OF REPORT =====\n")
            
            messagebox.showinfo("Export Successful", f"Statistics exported successfully to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export statistics: {str(e)}")
    
    def exit_application(self):
        """Clean exit of the application"""
        if self.is_running:
            self.toggle_monitoring()
        
        # Ask user if they want to save stats before exiting
        if self.monitoring_start_time is not None:
            if messagebox.askyesno("Save Statistics", "Would you like to export statistics before exiting?"):
                self.export_statistics()
        
        self.root.destroy()
    
    def run(self):
        """Run the application main loop"""
        self.root.mainloop()

class FatigueLogger:
    """A class to log and analyze fatigue patterns over time"""
    
    def __init__(self, log_file="fatigue_log.json"):
        self.log_file = log_file
        self.log_data = self.load_log()
    
    def load_log(self):
        """Load existing log data or create new"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, "r") as f:
                    return json.load(f)
            else:
                return {"sessions": []}
        except Exception:
            return {"sessions": []}
    
    def save_log(self):
        """Save log data to file"""
        with open(self.log_file, "w") as f:
            json.dump(self.log_data, f, indent=4, default=str)
    
    def add_session(self, session_data):
        """Add a new session to the log"""
        self.log_data["sessions"].append({
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "duration": str(session_data["end_time"] - session_data["start_time"]),
            "drowsy_episodes": session_data["drowsy_episodes"],
            "yawn_episodes": session_data["yawn_episodes"],
            "emergency_alerts": session_data["emergency_contacts"]
        })
        self.save_log()
    
    def get_weekly_summary(self):
        """Return a summary of fatigue patterns for the past week"""
        today = datetime.datetime.now().date()
        week_ago = today - datetime.timedelta(days=7)
        
        weekly_sessions = []
        for session in self.log_data["sessions"]:
            session_date = datetime.datetime.strptime(session["date"], "%Y-%m-%d").date()
            if week_ago <= session_date <= today:
                weekly_sessions.append(session)
        
        if not weekly_sessions:
            return "No sessions recorded in the past week."
        
        total_drowsy = sum(s["drowsy_episodes"] for s in weekly_sessions)
        total_yawns = sum(s["yawn_episodes"] for s in weekly_sessions)
        total_alerts = sum(s["emergency_alerts"] for s in weekly_sessions)
        
        return {
            "session_count": len(weekly_sessions),
            "total_drowsy_episodes": total_drowsy,
            "total_yawn_episodes": total_yawns,
            "total_emergency_alerts": total_alerts,
            "avg_drowsy_per_session": total_drowsy / len(weekly_sessions),
            "avg_yawns_per_session": total_yawns / len(weekly_sessions)
        }
    
    def generate_report(self, filename="fatigue_analysis_report.txt"):
        """Generate a comprehensive fatigue analysis report"""
        if not self.log_data["sessions"]:
            return "No session data available for analysis."
        
        weekly_summary = self.get_weekly_summary()
        
        with open(filename, "w") as f:
            f.write("===== FATIGUE ANALYSIS REPORT =====\n\n")
            f.write(f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("WEEKLY SUMMARY:\n")
            if isinstance(weekly_summary, dict):
                f.write(f"Sessions in the past week: {weekly_summary['session_count']}\n")
                f.write(f"Total drowsy episodes: {weekly_summary['total_drowsy_episodes']}\n")
                f.write(f"Total yawn episodes: {weekly_summary['total_yawn_episodes']}\n")
                f.write(f"Total emergency alerts: {weekly_summary['total_emergency_alerts']}\n")
                f.write(f"Average drowsy episodes per session: {weekly_summary['avg_drowsy_per_session']:.2f}\n")
                f.write(f"Average yawn episodes per session: {weekly_summary['avg_yawns_per_session']:.2f}\n\n")
            else:
                f.write(f"{weekly_summary}\n\n")
            
            f.write("ALL SESSIONS:\n")
            for idx, session in enumerate(self.log_data["sessions"], 1):
                f.write(f"Session {idx} - {session['date']} {session['time']}\n")
                f.write(f"  Duration: {session['duration']}\n")
                f.write(f"  Drowsy Episodes: {session['drowsy_episodes']}\n")
                f.write(f"  Yawn Episodes: {session['yawn_episodes']}\n")
                f.write(f"  Emergency Alerts: {session['emergency_alerts']}\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            # Add some simple recommendations based on the data
            total_sessions = len(self.log_data["sessions"])
            total_drowsy = sum(s["drowsy_episodes"] for s in self.log_data["sessions"])
            
            if total_drowsy / total_sessions > 5:
                f.write("- You appear to experience significant drowsiness. Consider improving your sleep schedule.\n")
            if total_drowsy / total_sessions > 10:
                f.write("- High frequency of drowsiness detected. Please consult with a healthcare professional.\n")
            
            f.write("\n===== END OF REPORT =====\n")
        
        return f"Report generated successfully: {filename}"

def main():
    """Main function to initialize and run the application"""
    root = tk.Tk()
    app = DrowsinessDetector(root)
    
    # Center the window on screen
    window_width = 1200
    window_height = 750
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    x_coordinate = int((screen_width / 2) - (window_width / 2))
    y_coordinate = int((screen_height / 2) - (window_height / 2))
    
    root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")
    
    # Add a logger instance to the app
    app.logger = FatigueLogger()
    
    # Override the exit function to log session data
    original_exit = app.exit_application
    
    def new_exit():
        if app.session_data["start_time"] is not None:
            if app.session_data["end_time"] is None:
                app.session_data["end_time"] = datetime.datetime.now()
            app.logger.add_session(app.session_data)
        original_exit()
    
    app.exit_application = new_exit
    
    # Run the application
    app.run()

if __name__ == "__main__":
    # Import json here as it was missing in the original imports
    import json
    main()