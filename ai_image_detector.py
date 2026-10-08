from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt

#For UI:
import customtkinter
from tkinter import *
from tkinter import filedialog
import tkinter as tk
from PIL import ImageTk, Image
import os
import shutil
import random
import string
from tkinter import messagebox


class AI_Detector:
    def __init__(self):
        self.aiDetector = load_model(os.path.join('models', 'aiDetectionV2.h5'))

    def detector(self, img_dir):
        img = cv2.imread(img_dir)
        resize = tf.image.resize(img, (256, 256))
        plt.imshow(resize.numpy().astype(int))

        yhat = self.aiDetector.predict(np.expand_dims(resize/255, 0))
        

        similarity_percentage, classification = calculate_similarity(yhat)

        plt.title(f"Predicted class is {classification}.")
        plt.xlabel("This model was trained for 50 epochs and got 92% accurracy.")
        plt.show()


def calculate_similarity(predicted_class):
    if predicted_class < 0.5:
        similarity_percentage = predicted_class
        classification = 'AI Generated'
    else:
        similarity_percentage = predicted_class
        classification = 'Real Image'

    return similarity_percentage, classification

# img_dir = 'real.png'
# detect = AI_Detector(img_dir)
# detect.detector()

def main():
    aiDetector = AI_Detector()

    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("blue")
    app = customtkinter.CTk()
    app.title("AI image DETECTOR")
    app.geometry("500x400")

    frame = customtkinter.CTkLabel(app, text="")
    frame.grid(row=0, column=0, sticky="w", padx=50, pady=20)

    def setPreviewPic(filepath):
        global img
        img = Image.open(filepath)
        img = img.resize((256, 256))
        img = ImageTk.PhotoImage(img)
        showPic = tk.Label(frame, bg="#1F6AA5", image=img)
        showPic.grid(row=1, column=0, columnspan=3, pady=5, ipady=0, sticky="nswe")
        pathEntry.insert(0, filepath)

    def selectPic():
        global filename
        filename = filedialog.askopenfilename(
            initialdir = os.getcwd(),
            title = "Select Image",
            filetypes = (("png images",".png"), ("jpg images", ".jpg"), ("jpeg images", ".jpeg"))
        )
        setPreviewPic(filename)

    def savePic():
        fileNameSplitted = filename.split(".")
        randomText = ''.join(random.choice(string.ascii_lowercase) for x in range(12))
        shutil.copy(filename, f"./images/{randomText}.{fileNameSplitted[1]}")
        setPreviewPic("./images/default.jpg") 
        messagebox.showinfo("Success", "Upload Successfully")
    
    def detectPic():
        aiDetector.detector(filename)

    selectBtn = customtkinter.CTkButton(frame, text="Browse Image", command=selectPic)
    pathEntry = customtkinter.CTkEntry(frame, width=200)
    saveBtn = customtkinter.CTkButton(frame, text="Detect", width=50, command=detectPic)
    showPic = tk.Label(frame, bg="#1F6AA5")

    setPreviewPic("./images/default.jpg") 

    selectBtn.grid(row=0, column=0, padx=1, pady=5, ipady=0, sticky="e")
    pathEntry.grid(row=0, column=1, padx=1, pady=5, ipady=0, sticky="e")
    saveBtn.grid(row=0, column=2, padx=1, pady=5, ipady=0, sticky="e")
    showPic.grid(row=1, column=0, columnspan=3, pady=5, ipady=0, sticky="nswe")


    app.resizable(False, False)
    app.mainloop()

main()
