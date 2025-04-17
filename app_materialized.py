import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import time
from pathlib import Path
import customtkinter as ctk

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MaterializedWasteClassifier:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Waste Classification")
        self.root.geometry("700x800")
        
        # Global variables
        self.model = None
        self.image_path = None
        self.labels = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
        
        # Theme setup
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        
        self.setup_ui()
        self.load_model()

    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def setup_ui(self):
        # Title
        self.title_label = ctk.CTkLabel(
            self.root,
            text="Waste Classification",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=20)

        # Description
        self.desc_label = ctk.CTkLabel(
            self.root,
            text="Upload an image to classify waste material",
            font=ctk.CTkFont(size=14)
        )
        self.desc_label.pack(pady=10)

        # Theme switcher
        self.theme_switch = ctk.CTkSwitch(
            self.root,
            text="Dark Mode",
            command=self.toggle_theme
        )
        self.theme_switch.pack(pady=10)
        if ctk.get_appearance_mode() == "Dark":
            self.theme_switch.select()

        # Image frame
        self.image_frame = ctk.CTkFrame(
            self.root,
            width=300,
            height=300
        )
        self.image_frame.pack(pady=20)
        self.image_frame.pack_propagate(False)

        self.display_label = ctk.CTkLabel(
            self.image_frame,
            text="Image will appear here"
        )
        self.display_label.pack(expand=True)

        # Buttons
        self.button_frame = ctk.CTkFrame(self.root)
        self.button_frame.pack(pady=20)

        self.upload_btn = ctk.CTkButton(
            self.button_frame,
            text="Upload Image",
            command=self.upload_image,
            font=ctk.CTkFont(size=14)
        )
        self.upload_btn.pack(side=tk.LEFT, padx=10)

        self.classify_btn = ctk.CTkButton(
            self.button_frame,
            text="Classify",
            command=self.classify_image,
            font=ctk.CTkFont(size=14)
        )
        self.classify_btn.pack(side=tk.LEFT, padx=10)

        # Results frame
        self.result_frame = ctk.CTkFrame(self.root)
        self.result_frame.pack(pady=20, padx=20, fill=tk.X)

        self.result_label = ctk.CTkLabel(
            self.result_frame,
            text="Classification Result: None",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.result_label.pack(pady=10)

        self.probability_label = ctk.CTkLabel(
            self.result_frame,
            text="Probability: 0%",
            font=ctk.CTkFont(size=14)
        )
        self.probability_label.pack(pady=5)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.root)
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10, padx=20, fill=tk.X)
        self.progress_bar.pack_forget()

        # Status bar
        self.status_label = ctk.CTkLabel(
            self.root,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=10)

    def toggle_theme(self):
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("Light")
        else:
            ctk.set_appearance_mode("Dark")

    def load_model(self):
        try:
            model_path = None
            possible_paths = [
                "trained_model.h5",
                os.path.join("models", "waste-classifier", "trained_model.h5"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.h5")
            ]

            for path in possible_paths:
                full_path = self.resource_path(path)
                if os.path.exists(full_path):
                    model_path = full_path
                    break

            if model_path is None:
                raise FileNotFoundError("Model file not found")

            self.status_label.configure(text="Loading model...")
            
            # Show progress
            self.progress_bar.pack(pady=10, padx=20, fill=tk.X)
            self.progress_bar.set(0)
            self.progress_bar.start()
            self.root.update()

            # Load model
            start_time = time.time()
            self.model = tf.keras.models.load_model(model_path, compile=False)
            load_time = time.time() - start_time

            # Warm up
            input_shape = self.model.input_shape[1:]
            dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
            self.model.predict(dummy_input, verbose=0)

            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            
            self.status_label.configure(
                text=f"Model loaded successfully in {load_time:.2f} seconds"
            )

        except Exception as e:
            if hasattr(self, 'progress_bar'):
                self.progress_bar.stop()
                self.progress_bar.pack_forget()
            self.status_label.configure(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            try:
                self.image_path = file_path
                
                # Display image
                img = Image.open(file_path)
                img = img.resize((250, 250), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)

                self.display_label.configure(image=img_tk, text="")
                self.display_label.image = img_tk

                # Reset results
                self.result_label.configure(text="Classification Result: None")
                self.probability_label.configure(text="Probability: 0%")
                self.status_label.configure(text=f"Loaded image: {Path(file_path).name}")

            except Exception as e:
                self.status_label.configure(text="Error loading image")
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def classify_image(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please upload an image first")
            return

        if not self.model:
            messagebox.showwarning("Warning", "Model not loaded")
            return

        try:
            # Show progress
            self.progress_bar.pack(pady=10, padx=20, fill=tk.X)
            self.progress_bar.set(0)
            self.progress_bar.start()
            
            self.status_label.configure(text="Classifying...")
            self.root.update()

            # Load and preprocess image
            img = Image.open(self.image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            input_shape = self.model.input_shape[1:3]  # Get expected height, width
            img_resized = img.resize(input_shape, Image.LANCZOS)
            img_array = np.array(img_resized, dtype=np.float32)
            
            # Ensure shape is (height, width, 3) and normalize
            if len(img_array.shape) == 2:  # Convert grayscale to RGB
                img_array = np.stack((img_array,) * 3, axis=-1)
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)

            # Predict
            start_time = time.time()
            pred = self.model.predict(img_batch, verbose=0)
            pred_time = time.time() - start_time

            # Get results
            pred_class_idx = np.argmax(pred[0])
            predicted_class = self.labels[pred_class_idx]
            probability = float(pred[0][pred_class_idx] * 100)

            # Update UI
            self.result_label.configure(text=f"Classification Result: {predicted_class}")
            self.probability_label.configure(text=f"Probability: {probability:.1f}%")
            self.status_label.configure(
                text=f"Classification complete in {pred_time:.2f} seconds"
            )

            # Show result dialog
            messagebox.showinfo(
                "Classification Result",
                f"The image is classified as: {predicted_class}\n"
                f"Probability: {probability:.2f}%"
            )

        except Exception as e:
            self.status_label.configure(text="Classification failed")
            messagebox.showerror("Error", f"Classification failed: {e}")
        finally:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()

    def run(self):
        # Set icon if available
        try:
            icon_path = self.resource_path("app_icon.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass
        
        self.root.mainloop()

if __name__ == "__main__":
    app = MaterializedWasteClassifier()
    app.run()