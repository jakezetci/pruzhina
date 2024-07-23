import detect_new
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

video_path = '498-498.MOV'

cam = cv.VideoCapture(filename=video_path, apiPreference=cv.CAP_FFMPEG)

max_frames = 100

ret, frame = cam.read()
N = 1
for i in range(max_frames):
    ret, new_frame = cam.read()
    new_frame = new_frame.astype(np.int16)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        # Our operations on the frame come here
    frame = np.sum([frame, new_frame], axis=0)
    N = N + 1
    # Display the resulting frame
    if cv.waitKey(1) == ord('q'):
        break
frame = frame / N
frame1 = frame.astype(np.int8)
#rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

plt.imshow(frame1)
plt.show()