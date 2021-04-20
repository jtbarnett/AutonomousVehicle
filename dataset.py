import cv2
import random
import numpy as np

frame = []
angle = []

# end of last batch pointers
train_batch_pointer = 0
val_batch_pointer = 0

# reads the data.txt file
with open("data/data.txt") as f:
    for line in f:
        frame.append("data/" + line.split()[0])
        angle.append(float(line.split()[1]) * 3.14159265 / 180)

# get number of frames
number_frames = len(frame)

# move all frames
c = list(zip(frame, angle))
random.shuffle(c)
frame, angle = zip(*c)

train_frame = frame[:int(len(frame) * 0.8)]
train_angle = angle[:int(len(frame) * 0.8)]

val_frame = frame[-int(len(frame) * 0.2):]
val_angle = angle[-int(len(frame) * 0.2):]

num_train_frames = len(train_frame)
num_val_frames = len(val_frame)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    frame_out = []
    angle_out = []
    for i in range(0, batch_size):
        frame_out.append(cv2.resize(cv2.imread(train_frame[(train_batch_pointer + i) % num_train_frames])[-150:], (200, 66)) / 255.0)
        angle_out.append([train_angle[(train_batch_pointer + i) % num_train_frames]])
    train_batch_pointer += batch_size
    return frame_out, angle_out

def LoadValBatch(batch_size):
    global val_batch_pointer
    frame_out = []
    angle_out = []
    for i in range(0, batch_size):
        frame_out.append(cv2.resize(cv2.imread(val_frame[(val_batch_pointer + i) % num_val_frames])[-150:], (200, 66)) / 255.0)
        angle_out.append([val_angle[(val_batch_pointer + i) % num_val_frames]])
    val_batch_pointer += batch_size
    return frame_out, angle_out
