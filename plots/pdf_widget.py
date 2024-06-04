import os
import tkinter as tk
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image, ImageTk

import tempfile
import fitz  # PyMuPDF

class ImageWidget:
    def __init__(self, frame_folder):
        self.frame_folder = frame_folder
        self.frame_files = sorted(file for file in os.listdir(frame_folder) if not file.startswith('.') and file.endswith(('.png', '.jpg', '.jpeg', '.pdf')))
        self.num_frames = len(self.frame_files)

        self.root = tk.Tk()

        self.frame_index = tk.IntVar()
        self.frame_index.set(0)

        self.frame_slider = tk.Scale(
            self.root,
            from_=0,
            to=self.num_frames - 1,
            orient=tk.HORIZONTAL,
            length=200,
            variable=self.frame_index,
            command=self.update_image_from_slider,
        )

        self.buttons_frame = tk.Frame(self.root)
        self.prev_button = tk.Button(self.buttons_frame, text="Previous Frame", command=self.prev_frame)
        self.next_button = tk.Button(self.buttons_frame, text="Next Frame", command=self.next_frame)

        self.label = tk.Label(self.root)
        self.label.pack(fill=tk.BOTH, expand=True)

        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.end_fullscreen)

        self.fullscreen = False



        # Initialize Tkinter image attribute
        self.tk_image = None
        self.update_image()

    def prev_frame(self):
        self.frame_index.set((self.frame_index.get() - 1) % self.num_frames)
        self.update_image()

    def next_frame(self):
        self.frame_index.set((self.frame_index.get() + 1) % self.num_frames)
        self.update_image()

    def update_image_from_slider(self, _):
        self.update_image()

    def update_image(self):
        current_frame_index = self.frame_index.get()
        frame_path = self.frame_folder / self.frame_files[current_frame_index]

        if frame_path.suffix.lower() == ".pdf":
            # Handle PDF file
            with fitz.open(frame_path) as pdf:
                # Get the first page of the PDF
                page = pdf[0]
                # Convert the PDF page to an image
                pixmap = page.get_pixmap()
                temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                pixmap._writeIMG(temp_image.name, "png", None)  # Use PNG format
                pil_image = Image.open(temp_image.name)
        else:
            # Handle image file
            pil_image = Image.open(frame_path)

        # Set fixed dimensions for the label
        label_width = 1000  # or any other fixed size you prefer
        label_height = 500  # or any other fixed size you prefer

        image_aspect_ratio = pil_image.width / pil_image.height
        if label_width / label_height > image_aspect_ratio:
            new_width = label_width
            new_height = int(label_width / image_aspect_ratio)
        else:
            new_width = int(label_height * image_aspect_ratio)
            new_height = label_height

        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        # Update Tkinter image
        self.tk_image = ImageTk.PhotoImage(resized_image)

        # Configure the label to display the image
        self.label.config(image=self.tk_image, width=label_width, height=label_height)

    def toggle_fullscreen(self, _=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)
        self.update_image()

    def end_fullscreen(self, _=None):
        self.fullscreen = False
        self.root.attributes("-fullscreen", False)
        self.update_image()

    def display(self):
        self.frame_slider.pack()
        self.buttons_frame.pack()
        self.prev_button.pack(side=tk.LEFT)
        self.next_button.pack(side=tk.LEFT)
        self.root.mainloop()

# Get the absolute path to the script's directory
script_directory = Path(__file__).parent

# Replace 'your_folder_path' with the actual path to the folder containing your PDF plots
# Join the script's directory with the relative path to your folder
folder_path = script_directory / "run_20240411_173103"

# Create the ImageWidget instance
image_widget = ImageWidget(folder_path)

# Display the widget with slider, "Previous Frame" button, and "Next Frame" button below the slider
image_widget.display()
