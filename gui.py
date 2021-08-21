from tkinter import *


curr_x, curr_y = 0, 0
data = []
color = 'black'

def locate_xy(event):
    global curr_x, curr_y
    curr_x, curr_y = event.x, event.y

def add_line(event):
    global curr_x, curr_y
    canvas.create_line((curr_x, curr_y, event.x, event.y), fill=color, width=5)
    curr_x, curr_y = event.x, event.y
    data.append((curr_x, curr_y))

def show_color(new_color):
    global color
    color = new_color

def new_canvas():
    canvas.delete('all')
    display_pallete()


window = Tk()

window.title("phiAI")
window.geometry("750x450")
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)

menubar = Menu(window)
window.config(menu=menubar)

submenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=submenu)
submenu.add_command(label='New Canvas', command=new_canvas)

canvas = Canvas(window, background='white')
canvas.grid(row=0, column=0)
canvas.config(width=280, height=280)
#canvas.create_line(20, 20, 80, 60)
canvas.bind('<Button-1>', locate_xy)
canvas.bind('<B1-Motion>', add_line)

def display_pallete():
    idC = canvas.create_rectangle((10, 10, 30, 30), fill='black')
    canvas.tag_bind(idC, '<Button-1>', lambda x: show_color('black'))
    idC = canvas.create_rectangle((10, 40, 30, 60), fill='ghost white')
    canvas.tag_bind(idC, '<Button-1>', lambda x: show_color('white'))


display_pallete()

window.mainloop()
