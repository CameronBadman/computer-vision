import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time


class ColorPickerGUI:
    def __init__(self, window, video_source=0):
        self.window = window
        self.window.title("Color Picker")

        # Initialize video capture
        print("Opening camera...")
        self.vid = cv2.VideoCapture(video_source)
        if self.vid.isOpened():
            print("Camera opened successfully")
            self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            print("Failed to open camera")
            return

        # Initialize variables
        self.last_click_pos = None
        self.selected_color = None
        self.is_running = True
        self.picked_colors = []

        # Create GUI
        self.create_widgets()

        # Start video thread
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Instructions
        ttk.Label(
            main_frame,
            text="Click on the video to sample colors. Press 'Save' to store a color.",
        ).pack(pady=5)

        # Video canvas
        self.canvas = tk.Canvas(main_frame, width=640, height=480)
        self.canvas.pack(padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Color info frame
        color_frame = ttk.Frame(main_frame)
        color_frame.pack(fill=tk.X, padx=5, pady=5)

        # Current color display
        self.color_label = ttk.Label(color_frame, text="Selected Color: None")
        self.color_label.pack(side=tk.LEFT, padx=5)

        # Color sample display (will be filled with color)
        self.color_sample = tk.Canvas(color_frame, width=50, height=25)
        self.color_sample.pack(side=tk.LEFT, padx=5)

        # Save button
        self.save_button = ttk.Button(
            color_frame, text="Save Color", command=self.save_color
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Saved colors list
        self.saved_colors_text = tk.Text(main_frame, height=5, width=50)
        self.saved_colors_text.pack(padx=5, pady=5)

    def on_canvas_click(self, event):
        if not hasattr(self, "current_frame"):
            return

        # Convert canvas coordinates to video coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        frame_height, frame_width = self.current_frame.shape[:2]

        # Scale coordinates
        x = int((event.x / canvas_width) * frame_width)
        y = int((event.y / canvas_height) * frame_height)

        # Get color at click position
        if 0 <= x < frame_width and 0 <= y < frame_height:
            bgr_color = self.current_frame[y, x]
            rgb_color = bgr_color[::-1]  # Convert BGR to RGB
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_color)
            hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]

            self.selected_color = {
                "hex": hex_color,
                "rgb": rgb_color,
                "hsv": hsv_color,
                "pos": (x, y),
            }

            # Update display
            self.color_label.config(text=f"Selected Color: {hex_color}")
            self.color_sample.delete("all")
            self.color_sample.create_rectangle(0, 0, 50, 25, fill=hex_color)

            print(
                f"Color at ({x}, {y}): RGB={rgb_color}, HSV={hsv_color}, HEX={hex_color}"
            )

    def save_color(self):
        if self.selected_color:
            color_info = (
                f"HEX: {self.selected_color['hex']}, "
                f"RGB: {tuple(self.selected_color['rgb'])}, "
                f"HSV: {tuple(self.selected_color['hsv'])}\n"
            )
            self.saved_colors_text.insert(tk.END, color_info)
            self.picked_colors.append(self.selected_color)

    def update(self):
        while self.is_running:
            ret, frame = self.vid.read()
            if ret:
                self.current_frame = frame

                # Draw crosshair at last click position
                if self.last_click_pos:
                    x, y = self.last_click_pos
                    cv2.drawMarker(frame, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

                # Convert frame for display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))

                # Update canvas
                self.window.after(0, self.update_canvas, photo)

            time.sleep(0.033)

    def update_canvas(self, photo):
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.photo = photo

    def __del__(self):
        self.is_running = False
        if hasattr(self, "vid"):
            self.vid.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorPickerGUI(root)
    root.mainloop()
