import cv2
from PIL import Image, ImageTk

def convert_frame(frame):
    return ImageTk.PhotoImage(
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    )