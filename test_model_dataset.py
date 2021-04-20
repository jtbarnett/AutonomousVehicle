import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import model
import cv2
from subprocess import call
import os

#check if on windows OS
windows = False
if os.name == 'nt':
    windows = True

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "save/model.ckpt")

wheel = cv2.imread('wheel2.png',0)
rows,columns = wheel.shape

smoothed_angle = 0

i = 0
while(cv2.waitKey(10) != ord('q')):
    full_frame = cv2.imread("data/" + str(i) + ".jpg")
    frame = cv2.resize(full_frame[-150:], (200, 66)) / 255.0
    angle = model.angle_output.eval(feed_dict={model.frame: [frame], model.keep_prob: 1.0})[0][0] * 180.0 / 3.14159265
    if not windows:
        call("clear")
    print("Predicted steering wheel angle: " + str(round(angle, 4)))
    cv2.imshow("Frames", full_frame)
    # change the angle of the steering wheel based off of the current angle and predicted steering wheel angle
    smoothed_angle += 0.2 * pow(abs((angle - smoothed_angle)), 2.0 / 3.0) * (angle - smoothed_angle) / abs(angle - smoothed_angle)
    M = cv2.getRotationMatrix2D((columns/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(wheel,M,(columns,rows))
    cv2.imshow("Wheel", dst)
    i += 1

cv2.destroyAllWindows()
