import cv2
import random
import numpy as np

# create the frame and angle numpy arrays
frame = []
angle = []

# end of last batch pointers
train_batch_pointer = 0
val_batch_pointer = 0

# reads the data.txt file
with open("data/data.txt") as f:
    for line in f:
        # add the current frame and angle data to the numpy arrays
        frame.append("data/" + line.split()[0])
        angle.append(float(line.split()[1]) * 3.14159265 / 180)

# get number of frames
number_frames = len(frame)

# move all frames
c = list(zip(frame, angle))
random.shuffle(c)
frame, angle = zip(*c)

# set up the train frame and angle
train_frame = frame[:int(len(frame) * 0.8)]
train_angle = angle[:int(len(frame) * 0.8)]

# set up the current value frame and angle
val_frame = frame[-int(len(frame) * 0.2):]
val_angle = angle[-int(len(frame) * 0.2):]

# get the number of trained frames and number of current value frames
num_train_frames = len(train_frame)
num_val_frames = len(val_frame)

# method for getting the current set of training data based on the input batch size
def LoadTrainBatch(batch_size):
    # get the train batch pointer
    global train_batch_pointer
    # create the output frame and anlge numpy arrays
    frame_out = []
    angle_out = []
    # append the current training data to the output frame and angle according the batch size
    for i in range(0, batch_size):
        frame_out.append(cv2.resize(cv2.imread(train_frame[(train_batch_pointer + i) % num_train_frames])[-150:], (200, 66)) / 255.0)
        angle_out.append([train_angle[(train_batch_pointer + i) % num_train_frames]])
    # update the train batch pointer
    train_batch_pointer += batch_size
    return frame_out, angle_out

# method for getting the current value data based on the input batch size
def LoadValBatch(batch_size):
    # get the value batch pointer
    global val_batch_pointer
    # create the output frame and anlge numpy arrays
    frame_out = []
    angle_out = []
    # append the current value data to the output frame and angle according the batch size
    for i in range(0, batch_size):
        frame_out.append(cv2.resize(cv2.imread(val_frame[(val_batch_pointer + i) % num_val_frames])[-150:], (200, 66)) / 255.0)
        angle_out.append([val_angle[(val_batch_pointer + i) % num_val_frames]])
    # update the value batch pointer
    val_batch_pointer += batch_size
    return frame_out, angle_out
