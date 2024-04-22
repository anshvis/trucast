from tkinter import *
import sys
print(sys.version)


root = Tk()

root.geometry("800x500")
root.title("TruCast")

label = Label(root, text="Hello World!", font=('Helvetica', 18))
label.pack()
root.update()  # Force the window to update


root.mainloop()