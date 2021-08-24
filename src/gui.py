from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from src.phiai.phiai import PhiAI


#trained = PhiAI([784, 10, 10])
trained = PhiAI([784, 20, 20, 10], lr=0.0985)
trained.load_model('res/trained/90.npz')

curr_x, curr_y = 0, 0
data = []
color = 'black'
canvas_w, canvas_h = 280, 280


def locate_xy(event):
    global curr_x, curr_y
    curr_x, curr_y = event.x, event.y

def add_line(event, width=int(canvas_w/28*2)):
    global curr_x, curr_y
    canvas.create_line((curr_x, curr_y, event.x, event.y), fill=color, smooth=1)
    curr_x, curr_y = event.x, event.y
    for i in range(width):
        data.append((int(curr_x-width/2+i), int(curr_y-width/2+i)))

def show_color(new_color):
    global color
    color = new_color

def new_canvas():
    global data
    canvas.delete('all')
    data = []
    display_pallete()

def extract_pixel_data():
    global canvas_w, canvas_h, data
    img = np.zeros((canvas_w, canvas_h))
    for i in range(len(data)):
        if data[i][0] < canvas_w and data[i][1] < canvas_h:
            img[data[i][0], data[i][1]] = 1
    return img

def input_data(plot=True, predict=True):
    grid_data = extract_pixel_data().T
    grid_data = cv2.resize(grid_data, (28, 28))

    if plot:
        plt.figure(figsize=(7, 7))
        plt.imshow(grid_data, interpolation='none', cmap='gray')
        plt.show()
    if predict:
        x_data = grid_data.reshape(grid_data.size, 1)
        trained.predict(x_data)
        predict = trained.argmax(trained.layers[trained.size - 1].output)
        print("Predicted: ", predict)

def display_pallete():
    idC = canvas.create_rectangle((10, 10, 30, 30), fill='black')
    canvas.tag_bind(idC, '<Button-1>', lambda x: show_color('black'))
    idC = canvas.create_rectangle((10, 40, 30, 60), fill='ghost white')
    canvas.tag_bind(idC, '<Button-1>', lambda x: new_canvas())
    idB = canvas.create_rectangle((10, 70, 30, 90), fill='gray')
    canvas.tag_bind(idB, '<Button-1>', lambda x: input_data())


window = Tk()
window.iconbitmap("res/cutie.ico")
window.title("phiAI")
window.geometry("750x450")
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)

menubar = Menu(window)
window.config(menu=menubar, bg='white smoke')
submenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=submenu)
submenu.add_command(label='New Canvas', command=new_canvas)

canvas = Canvas(window, background='white')
canvas.grid(row=0, column=0)
canvas.config(width=canvas_w, height=canvas_h)

canvas.bind('<Button-1>', locate_xy)
canvas.bind('<B1-Motion>', add_line)


display_pallete()
window.mainloop()
