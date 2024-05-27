import PIL
from PIL import ImageDraw
from tkinter import *
import numpy as np
import NN
from matplotlib import pyplot
import preprocess
from tkinter import messagebox


width = NN.ROWS * 25  # canvas width
height = NN.COLS * 25  # canvas height
black = 0  # background color
white = 255  # brush color 
brush_width = 30 
# def resize(image):
#     new_array = np.zeros((28,28))
#     for i in range(NN.ROWS):
#         for j in range(NN.COLS):
#             average = 0
#             for old_i in range(i * 10, i * 10 + 10):
#                 for old_j in range(j * 10, j * 10 + 10):
#                     average += image[old_i, old_j]
#             average /= 100
#             new_array[i, j] = average
#     return new_array


def save_image():
    # convert image to np array
    
    # resized = resize(image_array)

    # print(resized.shape)

    global output_image
    output_image_resized = output_image.resize((28, 28))
    image_array = np.array(output_image_resized)

    data = np.load('model.npy', allow_pickle=True).item()

    first_weights = data['first_weights']
    output_weights = data['output_weights']
    first_biases = data['first_biases']
    output_biases = data['output_biases']
    model = NN.Model(first_weights,
                  output_weights,
                  first_biases,
                  output_biases)
    hidden_activation, output_vector = NN.feed_forward(((np.array(image_array).flatten()).T) / 255, model)
    index_max = 0
    for j in range(1, len(output_vector)):
        if output_vector[j] > output_vector[index_max]:
            index_max = j
    messagebox.showinfo(f"{(NN.get_softmax_function(output_vector)[index_max] * 100)/ np.linalg.norm(NN.get_softmax_function(output_vector)):.2f}% confident", index_max)
    print()


    
    pyplot.imshow(image_array)
    pyplot.show()
    
    # close canvas
    master.destroy()

def paint(event):
    global last_x, last_y
    x1, y1 = last_x, last_y
    x2, y2 = event.x, event.y
    if last_x and last_y:
        canvas.create_line(x1, y1, x2, y2, fill="white", width=brush_width, capstyle=ROUND)
        draw.line([x1, y1, x2, y2], fill=white, width=brush_width)
    last_x, last_y = x2, y2

def set_last_point(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def reset_last_point(event):
    global last_x, last_y
    last_x, last_y = None, None

master = Tk()

# create a tkinter canvas
canvas = Canvas(master, width=width, height=height, bg='black')
canvas.pack()

# create pil image to draw on
output_image = PIL.Image.new("L", (width, height), black)
draw = ImageDraw.Draw(output_image)
canvas.pack(expand=YES, fill=BOTH)
canvas.bind("<B1-Motion>", paint)
canvas.bind("<Button-1>", set_last_point)
canvas.bind("<ButtonRelease-1>", reset_last_point)

button = Button(text="Done", command= save_image)
button.pack()


last_x, last_y = None, None

master.mainloop()