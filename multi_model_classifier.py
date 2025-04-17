import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from pathlib import Path
import customtkinter as ctk

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global variables
models = {}
image_path = None
labels = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']
model_results = {}
results_frame = None  # Global results_frame variable

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def setup_ui(root):
    global results_frame  # Declare results_frame as global

    root.title("Multi-Model Waste Classifier")
    root.geometry("800x800")
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")

    title_label = ctk.CTkLabel(root, text="Multi-Model Waste Classification",
                               font=ctk.CTkFont(size=24, weight="bold"))
    title_label.pack(pady=20)

    image_frame = ctk.CTkFrame(root, width=300, height=300)
    image_frame.pack(pady=20)
    display_label = ctk.CTkLabel(image_frame, text="Image will appear here")
    display_label.pack(expand=True)

    button_frame = ctk.CTkFrame(root)
    button_frame.pack(pady=20)

    upload_btn = ctk.CTkButton(button_frame, text="Upload Image",
                               command=lambda: upload_image(root, display_label))
    upload_btn.pack(side=tk.LEFT, padx=10)

    classify_btn = ctk.CTkButton(button_frame, text="Classify",
                                 command=lambda: classify_image(root, display_label))
    classify_btn.pack(side=tk.LEFT, padx=10)

    results_frame = ctk.CTkFrame(root)
    results_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

    progress_bar = ctk.CTkProgressBar(root)
    progress_bar.set(0)
    progress_bar.pack(pady=10, padx=20, fill=tk.X)
    progress_bar.pack_forget()

    status_label = ctk.CTkLabel(root, text="Ready")
    status_label.pack(pady=10)

    return display_label, results_frame, progress_bar, status_label

def load_all_models(root, results_frame, status_label):
    global models, model_results
    model_files = {
        'Base Model': 'trained_model.h5',
        'Alternative Model': 'Garbage.h5',
        # 'Fine-tuned Model': 'final_model_weights.hdf5'  # Commented out .hdf5 model
    }

    for model_name, model_file in model_files.items():
        try:
            # Try different possible paths
            possible_paths = [
                model_file,  # Current directory
                os.path.join(os.path.dirname(__file__), model_file),  # Same directory as script
                os.path.join('models', model_file),  # models subdirectory
                os.path.join('models', 'waste-classifier', model_file)  # nested models directory
            ]

            file_path = None
            for path in possible_paths:
                try:
                    full_path = resource_path(path)
                    if os.path.exists(full_path):
                        file_path = full_path
                        break
                except Exception:
                    continue

            if file_path is None:
                raise FileNotFoundError(f"Model file {model_file} not found in any of the expected locations")

            status_label.configure(text=f"Loading {model_name}...")
            root.update()

            model = tf.keras.models.load_model(file_path, compile=False)
            models[model_name] = model

            # Create result labels for this model
            model_frame = ctk.CTkFrame(results_frame)
            model_frame.pack(pady=5, fill=tk.X)

            ctk.CTkLabel(model_frame, text=model_name,
                        font=ctk.CTkFont(weight="bold")).pack()

            result_label = ctk.CTkLabel(model_frame, text="Result: None")
            result_label.pack()

            confidence_label = ctk.CTkLabel(model_frame, text="Confidence: 0%")
            confidence_label.pack()

            model_results[model_name] = {
                'frame': model_frame,
                'result': result_label,
                'confidence': confidence_label
            }
        except Exception as e:
            messagebox.showwarning("Model Loading Warning",
                                 f"Failed to load {model_name}: {str(e)}")

    if not models:
        messagebox.showerror("Error", "No models could be loaded!")
        root.quit()
    else:
        status_label.configure(text=f"Loaded {len(models)} models successfully")

def upload_image(root, display_label):
    global image_path, model_results
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if file_path:
        try:
            image_path = file_path

            img = Image.open(file_path)
            img = img.resize((250, 250), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            display_label.configure(image=img_tk, text="")
            display_label.image = img_tk

            for model_name in models:
                model_results[model_name]['result'].configure(text="Result: None")
                model_results[model_name]['confidence'].configure(text="Confidence: 0%")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

def classify_image(root, display_label):
    global image_path, models, model_results, results_frame
    if not image_path:
        messagebox.showwarning("Warning", "Please upload an image first")
        return

    if not models:
        messagebox.showwarning("Warning", "No models available")
        return

    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        ensemble_predictions = []

        for model_name, model in models.items():
            input_shape = model.input_shape[1:3]  # Get expected height, width
            img_resized = img.resize(input_shape, Image.LANCZOS)
            img_array = np.array(img_resized, dtype=np.float32)

            # Ensure shape is (height, width, 3) and normalize
            if len(img_array.shape) == 2:  # Convert grayscale to RGB
                img_array = np.stack((img_array,) * 3, axis=-1)
            img_array = img_array / 255.0

            # Add batch dimension
            img_batch = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_batch, verbose=0)
            pred_class_idx = np.argmax(pred[0])
            confidence = float(pred[0][pred_class_idx] * 100)

            ensemble_predictions.append(pred[0])

            model_results[model_name]['result'].configure(
                text=f"Result: {labels[pred_class_idx]}")
            model_results[model_name]['confidence'].configure(
                text=f"Confidence: {confidence:.1f}%")

        ensemble_pred = np.mean(ensemble_predictions, axis=0)
        ensemble_class_idx = np.argmax(ensemble_pred)
        ensemble_confidence = float(ensemble_pred[ensemble_class_idx] * 100)

        if 'Ensemble' not in model_results:
            ensemble_frame = ctk.CTkFrame(results_frame)
            ensemble_frame.pack(pady=10, fill=tk.X)

            ctk.CTkLabel(ensemble_frame, text="Ensemble Prediction",
                         font=ctk.CTkFont(size=16, weight="bold")).pack()

            result_label = ctk.CTkLabel(ensemble_frame,
                                        text=f"Final Result: {labels[ensemble_class_idx]}")
            result_label.pack()

            confidence_label = ctk.CTkLabel(ensemble_frame,
                                            text=f"Confidence: {ensemble_confidence:.1f}%")
            confidence_label.pack()

            model_results['Ensemble'] = {
                'frame': ensemble_frame,
                'result': result_label,
                'confidence': confidence_label
            }
        else:
            model_results['Ensemble']['result'].configure(
                text=f"Final Result: {labels[ensemble_class_idx]}")
            model_results['Ensemble']['confidence'].configure(
                text=f"Confidence: {ensemble_confidence:.1f}%")

    except Exception as e:
        messagebox.showerror("Error", f"Classification failed: {e}")

def main():
    root = ctk.CTk()
    display_label, results_frame, progress_bar, status_label = setup_ui(root)
    load_all_models(root, results_frame, status_label)
    root.mainloop()

if __name__ == "__main__":
    main()
