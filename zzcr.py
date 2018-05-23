# -*-coding:utf-8-*
"""
Main module, this file contains the GUI
"""

import sys
import os
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import Image, ImageTk

from imageconvert import isPictureFile
from extractor import char_detector


class Interface(Frame):
    """
    The thing to display
    """
    def __init__(self, window, width=800, height=600, **kwargs):
        """
        Sets up all the widgets in the window
        """
        Frame.__init__(self, window, width=width, height=height, **kwargs)
        self.pack(fill=BOTH)
        # Create the frames
        self.frame_select = Frame(self)
        self.frame_convert = Frame(self)
        self.frame_picture_text = Frame(self)
        # Create the widgets
        self.label_welcome = Label(self, text="Welcome to ZZCR, the character recognition software made by ZZs")
        self.label_select = Label(self.frame_select, text="Select a picture")
        self.label_convert = Label(self.frame_convert, text="Then convert this picture")
        self.str_path_to_picture = StringVar()
        self.entry_path = Entry(self.frame_select, textvariable=self.str_path_to_picture, width=50)
        self.button_browse = Button(self.frame_select, text="Browse", command=self.browse)
        self.button_convert = Button(self.frame_convert, text="Convert to text", command=self.convert)
        self.canvas_img = Canvas(self.frame_picture_text)
        self.img = self.canvas_img.create_image(0,0, anchor=NW)
        self.text_converted_text = Text(self.frame_picture_text, width=40)
        # Display the widgets in the window
        self.label_welcome.pack(side="top", fill=X)
        self.frame_select.pack(side="top", fill=X)
        self.label_select.pack(side="left")
        self.entry_path.pack(side="left")
        self.button_browse.pack(side="left")
        self.frame_convert.pack(side="top", fill=X)
        self.label_convert.pack(side="left")
        self.button_convert.pack(side="left")
        self.frame_picture_text.pack(side="top", fill=X)
        self.canvas_img.pack(side="left", fill=Y)
        self.text_converted_text.pack(side="left", fill=Y)

    def browse(self):
        """
        This function is called when the user click the browse button.
        It open a dialog box to help him select a file.
        """
        filename = filedialog.askopenfilename(initialdir=".", title="Select a picture")
        self.str_path_to_picture.set(filename)
        if(not isPictureFile(filename)):
            messagebox.showerror("Error", "The file %s is not a picture" % filename, parent=self)
            return None
        img = Image.open(filename)
        img = img.resize((400, 400*img.size[1]//img.size[0]))
        self.canvas_img["width"] = img.size[0]
        self.canvas_img["height"] = img.size[1]
        self.imgtk = ImageTk.PhotoImage(img) # Warning: we must keep a reference on the PhotoImage otherwise it is destroyed and nothing is displayed
        self.canvas_img.itemconfig(self.img, image=self.imgtk)

    def convert(self):
        """
        This function is called when the user click the convert button.
        It runs the recognition function with the file in the entry.
        """
        filename = self.str_path_to_picture.get()
        if(not isPictureFile(filename)):
            messagebox.showerror("Error", "The file %s is not a picture" % filename, parent=self)
            return None
        text_recognized = recognition(filename)
        self.text_converted_text.delete("1.0", END)
        self.text_converted_text.insert("1.0", text_recognized)
        

def gui():
    """
    This function display the GUI.
    """
    # Create the window and its contents
    window = Tk()
    window.title("ZZCR")
    window.iconbitmap("icon.ico")
    inter = Interface(window)
    # Show the window
    inter.mainloop()


def recognition(picture_filename):
    """
    This function calls the other modules
    """
    out = ""
    try:
        out = char_detector(picture_filename)
    except IOError:
        out = "Error: The file %s is not a picture" % picture_filename
        print(out, file=sys.stderr)
    return out


if __name__ == "__main__":
    if(len(sys.argv) == 1):
        gui()
    elif(len(sys.argv) == 2):
        if(os.path.isfile(sys.argv[1]) and isPictureFile(sys.argv[1])):
            text = recognition(sys.argv[1])
            print(text)
        else:
            print("Error: Cannot read the file %s, this is not a readable picture" % sys.argv[1], file=sys.stderr)
    else:
        print("Help:\nRun the GUI: no argument\nRun the recognition program only: one argument = path_to_the_picture")
