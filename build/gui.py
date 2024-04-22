
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"/Users/anshviswanathan/Documents/GitHub/trucast/build/assets/frame0")

def import_file():
    # This function will be called when the import button is clicked
    filepath = filedialog.askopenfilename()  # Opens a dialog to select a file
    if filepath:
        print("File selected:", filepath)
        # You can now do something with the selected file path

def export_path():
    # This function will be called when the export button is clicked
    directory = filedialog.askdirectory()  # Opens a dialog to select a directory
    if directory:
        print("Directory selected:", directory)
        # You can now do something with the selected directory


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("700x500")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 500,
    width = 700,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_rectangle(
    0.0,
    0.0,
    430.0,
    500.0,
    fill="#04B540",
    outline="")

canvas.create_text(
    58.0,
    156.0,
    anchor="nw",
    text="Welcome to TruCast!",
    fill="#FFFFFF",
    font=("InriaSans Regular", 35 * -1)
)


# Create a Text widget instead of using create_text
text_box = Text(
    window,
    bg="#04B540",
    fg="#FFFFFF",
    font=("Helvetica", 17),
    width=40,  # Width in characters, adjust as needed
    height=10,  # Height in lines, adjust as needed
    wrap="word",  # Wrap text at word boundaries
    bd=0,  # No border
    highlightthickness=0,  # No focus highlight
)

# Insert the desired text
text_box.insert("1.0", "TruCast is an application built to help forecast revenue for TruBridge. "
                        "It utilizes modern statistical and machine learning models to predict revenue, "
                        "such as ARIMA and Facebook’s PROPHET model.")

# Place the Text widget on the canvas
text_box.place(x=23, y=273)

# Disable the text box if you don't want it to be editable
text_box.configure(state="disabled")

canvas.create_text(
    445.0,
    151.0,
    anchor="nw",
    text="Import financial data",
    fill="#000000",
    font=("InriaSans Bold", 25 * -1)
)

canvas.create_text(
    444.0,
    248.0,
    anchor="nw",
    text="Import file",
    fill="#000000",
    font=("InriaSans Bold", 17 * -1)
)

import_button = Button(window, text="Choose file...", command=import_file)
import_button_window = canvas.create_window(444.0, 274.0, anchor="nw", window=import_button)

canvas.create_text(
    444.0,
    362.0,
    anchor="nw",
    text="Export path",
    fill="#000000",
    font=("InriaSans Bold", 17 * -1)
)


export_button = Button(window, text="Export path", command=export_path, bd=0, highlightthickness=0, relief='flat')
export_button_window = canvas.create_window(458.0, 388.0, anchor="nw", window=export_button)


window.resizable(False, False)
window.mainloop()
