import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time


class ColorSquareDetectorGUI:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Color Square Detector")

        # Initialize variables first
        self.hex_color = "#E4ECD7"  # Color from picker
        self.base_hsv = np.array([41, 23, 236])  # HSV values from picker
        self.is_running = True
        self.detected_distance = "No squares detected"
        self.pixels_per_cm = None  # Calibration value
        self.reference_distance_cm = 7.6  # Standard post-it note width in cm

        # Initialize video source
        print(f"Trying to open camera {video_source}...")
        self.vid = cv2.VideoCapture(video_source)

        if self.vid.isOpened():
            print("Successfully opened camera")
            self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.vid.get(cv2.CAP_PROP_FPS)
            print(f"Camera properties: {width}x{height} @ {fps}fps")
        else:
            print("Failed to open camera")
            return

        # Create GUI elements
        self.create_widgets()

        # Start video thread
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def create_widgets(self):
        # Main container
        main_container = ttk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Control frame at top
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Color info
        ttk.Label(control_frame, text="Target Color:").pack(side=tk.LEFT, padx=5)
        color_display = tk.Canvas(control_frame, width=30, height=20)
        color_display.pack(side=tk.LEFT, padx=5)
        color_display.create_rectangle(0, 0, 30, 20, fill=self.hex_color)

        # Status labels
        self.squares_label = ttk.Label(control_frame, text="Squares: 0")
        self.squares_label.pack(side=tk.RIGHT, padx=5)
        self.distance_label = ttk.Label(control_frame, text="Distance: ---")
        self.distance_label.pack(side=tk.RIGHT, padx=5)

        # Video frames container
        video_container = ttk.Frame(main_container)
        video_container.pack(fill=tk.BOTH, expand=True)

        # Main video canvas (left)
        self.canvas = tk.Canvas(video_container, width=640, height=480)
        self.canvas.pack(side=tk.LEFT, padx=5, pady=5)

        # Mask preview canvas (right)
        self.mask_canvas = tk.Canvas(video_container, width=320, height=240)
        self.mask_canvas.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Label(video_container, text="Detection Mask").pack(side=tk.LEFT)

    def hex_to_hsv_range(self, hex_color):
        """Convert hex color to HSV range with proper numpy types"""
        # Use pre-defined HSV values from color picker with wider ranges
        h, s, v = self.base_hsv

        # Much wider tolerances for pastel colors
        h_tolerance = 20  # Wider hue range
        s_tolerance = 50  # Much wider saturation range for pastel colors
        v_tolerance = 70  # Very wide value range for different lighting

        # Calculate ranges with proper types
        lower_hsv = np.array(
            [
                max(0, h - h_tolerance),
                max(0, 0),  # Allow very low saturation
                max(0, v - v_tolerance),
            ],
            dtype=np.uint8,
        )

        upper_hsv = np.array(
            [
                min(180, h + h_tolerance),
                min(255, 100),  # Cap saturation at moderate level
                min(255, 255),  # Allow full brightness range
            ],
            dtype=np.uint8,
        )

        print(f"HSV Ranges - Lower: {lower_hsv}, Upper: {upper_hsv}")  # Debug info

        return lower_hsv, upper_hsv

    def detect_squares(self, frame):
        """Detect squares in frame"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get color range
        lower_color, upper_color = self.hex_to_hsv_range(self.hex_color)

        # Create mask
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Create mask preview
        mask_small = cv2.resize(mask, (320, 240))
        mask_rgb = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2RGB)
        self.mask_photo = ImageTk.PhotoImage(image=Image.fromarray(mask_rgb))
        self.mask_canvas.create_image(0, 0, image=self.mask_photo, anchor=tk.NW)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        squares = []
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 1000:  # Minimum area threshold
                continue

            # Approximate shape
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.1 * peri, True)  # More forgiving

            # Accept shapes with 4-6 points
            if 4 <= len(approx) <= 6:
                squares.append(approx)
                print(f"Found shape with area: {area}, points: {len(approx)}")

        return squares

    def calibrate(self):
        """Calibrate the measurement system using a single post-it note"""
        ret, frame = self.vid.read()
        if not ret:
            print("Could not read frame for calibration")
            return

        squares = self.detect_squares(frame)
        if len(squares) >= 1:
            # Get the width of the first square in pixels
            rect = cv2.minAreaRect(squares[0])
            width_px = min(rect[1])  # Get the smaller dimension

            # Calculate pixels per cm using known post-it width
            self.pixels_per_cm = width_px / self.reference_distance_cm
            self.calib_label.config(text=f"Calibrated: {self.pixels_per_cm:.1f} px/cm")
            print(f"Calibrated: {self.pixels_per_cm:.1f} pixels per cm")
        else:
            print("No squares detected for calibration")

    def calculate_distance(self, square1, square2):
        """Calculate distance between centers of squares"""
        center1 = np.mean(square1, axis=0)[0]
        center2 = np.mean(square2, axis=0)[0]

        # Calculate distance in pixels
        pixel_distance = np.sqrt(
            ((center1[0] - center2[0]) ** 2) + ((center1[1] - center2[1]) ** 2)
        )

        # Convert to cm if calibrated
        if self.pixels_per_cm is not None:
            return pixel_distance / self.pixels_per_cm
        return pixel_distance

    def update(self):
        """Main update loop"""
        while self.is_running:
            ret, frame = self.vid.read()
            if not ret:
                continue

            # Process frame
            squares = self.detect_squares(frame)

            # Update squares label
            self.window.after(
                0, self.squares_label.config, {"text": f"Squares: {len(squares)}"}
            )

            # Draw detected squares
            if len(squares) >= 2:
                # Draw squares
                cv2.drawContours(frame, squares, -1, (0, 255, 0), 2)

                # Calculate and draw distance
                distance = self.calculate_distance(squares[0], squares[1])
                center1 = np.mean(squares[0], axis=0)[0].astype(int)
                center2 = np.mean(squares[1], axis=0)[0].astype(int)

                # Draw line between centers
                cv2.line(frame, tuple(center1), tuple(center2), (255, 0, 0), 2)

                # Update distance label
                self.window.after(
                    0, self.distance_label.config, {"text": f"Distance: {distance:.1f}"}
                )

            # Convert frame for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))

            # Update canvas
            self.window.after(0, self.update_canvas)

            time.sleep(0.033)

    def update_canvas(self):
        """Update the main video canvas"""
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def __del__(self):
        """Cleanup"""
        self.is_running = False
        if hasattr(self, "vid"):
            self.vid.release()


if __name__ == "__main__":
    root = tk.Tk()
    # Try different camera indices
    for camera_index in [0, 2, 1]:
        print(f"\nTrying camera index {camera_index}")
        app = ColorSquareDetectorGUI(root, video_source=camera_index)
        if hasattr(app, "vid") and app.vid is not None and app.vid.isOpened():
            break
    root.mainloop()
