import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import time

# Set environment variables to avoid errors and reduce warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR

# Global variables
root = None
image_path = None
model = None
display_label = None
result_label = None
probability_label = None
status_bar = None
progress_bar = None
progress_frame = None

# Labels for classification
labels = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def load_model():
    global model, status_bar, progress_bar, root
    try:
        # Try to find the model file
        model_path = None
        possible_paths = [
            "trained_model.h5",
            os.path.join("models", "waste-classifier", "trained_model.h5"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.h5")
        ]

        for path in possible_paths:
            full_path = resource_path(path)
            if os.path.exists(full_path):
                model_path = full_path
                break

        if model_path is None:
            raise FileNotFoundError("Model file not found")

        status_bar.config(text="Loading model...")

        # Show progress bar during loading
        progress_bar.pack(fill=tk.X, expand=True)
        progress_bar.start(10)
        root.update()

        # Load the model with timing
        start_time = time.time()
        model = tf.keras.models.load_model(model_path, compile=False)
        load_time = time.time() - start_time

        # Warm up the model with a dummy prediction
        input_shape = model.input_shape[1:]
        dummy_input = np.zeros((1,) + input_shape, dtype=np.float32)
        model.predict(dummy_input, verbose=0)

        # Hide progress bar
        progress_bar.stop()
        progress_bar.pack_forget()

        status_bar.config(text=f"Model loaded successfully in {load_time:.2f} seconds. Ready to classify images.")
        root.update()

    except Exception as e:
        if progress_bar:
            progress_bar.stop()
            progress_bar.pack_forget()
        status_bar.config(text=f"Error: {str(e)}")
        messagebox.showerror("Error", f"Failed to load model: {e}")

def upload_image():
    global image_path, display_label, status_bar, result_label, probability_label
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if file_path:
        try:
            image_path = file_path
            status_bar.config(text=f"Image loaded: {os.path.basename(file_path)}")

            # Display the image
            img = Image.open(file_path)
            img = img.resize((250, 250), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            display_label.config(image=img_tk, text="")
            display_label.image = img_tk  # Keep a reference

            # Reset result
            result_label.config(text="Classification Result: None")
            probability_label.config(text="Probability: 0%")

        except Exception as e:
            status_bar.config(text="Error loading image")
            messagebox.showerror("Error", f"Failed to load image: {e}")

def classify_image():
    global image_path, model, status_bar, result_label, probability_label, root, progress_bar
    if image_path is None:
        messagebox.showwarning("Warning", "Please upload an image first")
        return

    if model is None:
        messagebox.showwarning("Warning", "Model not loaded")
        return

    # Show progress bar
    progress_bar.pack(fill=tk.X, expand=True)
    progress_bar.start(10)

    try:
        status_bar.config(text="Classifying...")
        root.update()

        # Load and preprocess the image
        with Image.open(image_path) as img:
            # Get the input shape from the model
            input_shape = model.input_shape[1:3]  # (height, width)
            img = img.resize((input_shape[1], input_shape[0]), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)

        # Make prediction with timing
        start_time = time.time()
        pred = model.predict(img_batch, verbose=0)
        pred_time = time.time() - start_time

        # Get the class with highest probability
        predicted_class_idx = np.argmax(pred[0])
        predicted_class = labels[predicted_class_idx]

        # Get the probability
        probability = float(pred[0][predicted_class_idx] * 100)

        # Update result
        result_label.config(text=f"Classification Result: {predicted_class}")
        probability_label.config(text=f"Probability: {probability:.1f}%")
        status_bar.config(text=f"Classification complete in {pred_time:.2f} seconds")

        # Show result in a message box
        messagebox.showinfo("Classification Result",
                           f"The image is classified as: {predicted_class}\nProbability: {probability:.2f}%")

    except Exception as e:
        status_bar.config(text="Classification failed")
        messagebox.showerror("Error", f"Classification failed: {e}")
    finally:
        # Hide and stop progress bar
        progress_bar.stop()
        progress_bar.pack_forget()

def create_ui():
    global root, display_label, result_label, probability_label, status_bar, progress_bar, progress_frame

    # Set background color
    root.configure(bg="#f0f0f0")

    # Title
    title_label = tk.Label(root, text="Waste Classification", font=("Arial", 24, "bold"), bg="#f0f0f0")
    title_label.pack(pady=20)

    # Description
    desc_label = tk.Label(root, text="Upload an image to classify waste material",
                         font=("Arial", 12), bg="#f0f0f0")
    desc_label.pack(pady=10)

    # Frame for image display
    image_frame = tk.Frame(root, bg="#ffffff", width=300, height=300, bd=2, relief=tk.GROOVE)
    image_frame.pack(pady=20)
    image_frame.pack_propagate(False)

    # Default image display
    display_label = tk.Label(image_frame, bg="#ffffff", text="Image will appear here")
    display_label.pack(fill=tk.BOTH, expand=True)

    # Button frame
    button_frame = tk.Frame(root, bg="#f0f0f0")
    button_frame.pack(pady=20)

    # Upload button
    upload_btn = tk.Button(button_frame, text="Upload Image", command=upload_image,
                          font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
    upload_btn.grid(row=0, column=0, padx=10)

    # Classify button
    classify_btn = tk.Button(button_frame, text="Classify", command=classify_image,
                            font=("Arial", 12), bg="#2196F3", fg="white", padx=10, pady=5)
    classify_btn.grid(row=0, column=1, padx=10)

    # Result frame
    result_frame = tk.Frame(root, bg="#f0f0f0")
    result_frame.pack(pady=10)

    # Result label
    result_label = tk.Label(result_frame, text="Classification Result: None",
                           font=("Arial", 14, "bold"), bg="#f0f0f0")
    result_label.pack()

    # Probability label
    probability_label = tk.Label(result_frame, text="Probability: 0%",
                                font=("Arial", 12), bg="#f0f0f0")
    probability_label.pack(pady=5)

    # Progress frame
    progress_frame = tk.Frame(root, bg="#f0f0f0")
    progress_frame.pack(pady=5, fill=tk.X, padx=20)
    progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="indeterminate")

    # Status bar
    status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # Update global variables
    globals()['display_label'] = display_label
    globals()['result_label'] = result_label
    globals()['probability_label'] = probability_label
    globals()['status_bar'] = status_bar
    globals()['progress_bar'] = progress_bar
    globals()['progress_frame'] = progress_frame

def main():
    global root
    try:
        # Initialize the main window
        root = tk.Tk()
        root.title("Waste Classification")
        root.geometry("700x600")

        # Set application icon if available
        try:
            icon_path = resource_path("app_icon.ico")
            if os.path.exists(icon_path):
                root.iconbitmap(icon_path)
        except Exception:
            # Continue without icon if there's an issue
            pass

        # Create UI and load model
        create_ui()
        load_model()

        # Start the main loop
        root.mainloop()
    except Exception as e:
        # Handle any unexpected errors during startup
        messagebox.showerror("Startup Error", f"An error occurred during application startup: {e}")
        if root:
            root.destroy()

if __name__ == "__main__":
    main()
