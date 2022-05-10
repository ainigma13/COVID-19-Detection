# import the necessary packages
# from pyimagesearch.gradcam import GradCAM
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import imutils
import cv2
import tkinter
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained model")
args = vars(ap.parse_args())

# load the pre-trained network
print("[INFO] loading pre-trained network...")
global model
model = load_model(args["model"])


def ExitApplication():
    MsgBox = tkinter.messagebox.askquestion('Exit Application', 'Are you sure you want to exit the application',
                                            icon='warning')
    if MsgBox == 'yes':
        root.destroy()
    else:
        tkinter.messagebox.showinfo('Return', 'You will now return to the application screen')


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.top1 = tkinter.Frame(window, borderwidth=5, relief='ridge', width=550, height=150, bg='black')
        self.top1.pack(side='top', anchor='n')
        self.canvas1 = tkinter.Canvas(self.top1, width=550, height=150, bg='slate blue')
        self.canvas1.pack(side='top', anchor='n', expand=True)
        self.cv_img = cv2.imread("1.png")
        self.cv_img = cv2.resize(self.cv_img, (150, 150))
        self.height, self.width, self.no_channels = self.cv_img.shape
        self.ph = ImageTk.PhotoImage(master=self.canvas1, image=Image.fromarray(self.cv_img))
        self.canvas1.create_image(0, 0, image=self.ph, anchor='nw')  # 0 0 are coordinate
        self.deptlabel1 = tkinter.Label(self.top1, bg='slate blue', justify=tkinter.CENTER, text='WELCOME ',
                                        font=('Times', 14, 'bold'))
        self.deptlabel1.config(foreground='snow')
        self.canvas1.create_window(325, 25, window=self.deptlabel1)
        self.deptlabel2 = tkinter.Label(self.top1, bg='slate blue', justify=tkinter.CENTER,
                                        text='COVID-19 DETECTION', font=('Times', 14, 'bold'))
        self.deptlabel2.config(foreground='snow')
        self.canvas1.create_window(325, 75, window=self.deptlabel2)
        self.iitlabel1 = tkinter.Label(self.top1, bg='slate blue', justify=tkinter.CENTER,
                                       text='USING CHEST X-RAYS', font=('Times', 14, 'bold'))
        self.iitlabel1.config(foreground='snow')
        self.canvas1.create_window(325, 125, window=self.iitlabel1)

        self.bottom = tkinter.Frame(window, borderwidth=5, relief='ridge', width=550, height=300, bg='black')
        self.bottom.pack(side='top', anchor='n')
        self.canvas2 = tkinter.Canvas(self.bottom, width=550, height=300, bg='lavender')
        self.canvas2.pack(side='top', anchor='n', expand=True)
        self.heading1 = tkinter.Label(self.bottom, bg='lavender', justify=tkinter.CENTER, relief='flat',
                                      text='Image selection ', font=('Times', 14, 'bold'))
        self.heading1.config(foreground='slate blue')
        self.canvas2.create_window(300, 16, window=self.heading1)  # coordinate wrt this frame
        self.btn_select = tkinter.Button(self.bottom, text="Load", borderwidth=2, relief='raised',
                                         font=('Times', 12, 'bold'), bg='green', width=15, command=self.select_image)
        self.canvas2.create_window(150, 250, window=self.btn_select)
        self.btn_exit = tkinter.Button(self.bottom, text='Exit', borderwidth=2, relief='raised',
                                       font=('Times', 12, 'bold'), command=ExitApplication, bg='red', width=15)
        self.canvas2.create_window(385, 250, window=self.btn_exit)

        # self.window.mainloop()

    def select_image(self):
        self.filename = filedialog.askopenfilename(initialdir="/home/dhrd/DhRD_Files/VENV_IRIS/keras-covid-19",
                                                   title="Select file", filetypes=(
            ("all files", "*.*"), ("png files", "*.png*"), ("jpeg files", "*.jpg")))

        self.orig = cv2.imread(self.filename)
        self.img = cv2.cvtColor(self.orig, cv2.COLOR_BGR2RGB)
        self.img = cv2.resize(self.img, (224, 224))
        self.img = self.img.astype("float") / 255.0
        self.img = img_to_array(self.img)
        self.img = np.expand_dims(self.img, axis=0)
        self.pred = model.predict(self.img)
        self.pred = self.pred.argmax(axis=1)[0]
        self.label = "COVID-19 Positive" if self.pred == 0 else "COVID-19 negative"
        self.color = (0, 0, 255) if self.pred == 0 else (0, 255, 0)
        self.orig = cv2.resize(self.orig, (224, 224))
        cv2.putText(self.orig, self.label, (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
        cv2.imshow("Result", self.orig)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # self.image=Image.open(self.filename)
        # self.size=(224,224)
        # self.resize_image=self.image.resize(self.size)
        # self.photo=ImageTk.PhotoImage(self.orig)
        # self.cv = tkinter.Canvas(width = 300,height = 300,bg = '#6699ff')
        # self.cv.pack(side='top', fill='both', expand='yes')
        # self.cv.create_image(10, 10, image=self.photo, anchor='nw')


# Create a window and pass it to the Application object

# App(tkinter.Tk(), "COVID19 detection")
root = tkinter.Tk()
root.geometry('550x450')
root.resizable(0, 0)
app = App(root, "COVID-19 detection")
root.mainloop()

