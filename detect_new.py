# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
from collections import namedtuple
import multiprocessing as mp
import cv2 as cv

shared_data_class = namedtuple(
    'shared_data_class', ['cap', 'color1', 'color2'])

def get_coors(frame, __color1, __color2, debug=False):
    indeces_1 = find_color(frame, __color1, 500, debug)
    indeces_2 = find_color(frame, __color2, 2000, debug)
    if 0 in np.shape(indeces_1):
        indeces_1 = np.array([[-10000, 10000], [-10000, 10000]])
    if 0 in np.shape(indeces_2):
        indeces_2 = np.array([[-10000, 10000], [-10000, 10000]])
    center1 = np.mean(indeces_1, axis=1)
    center2 = np.mean(indeces_2, axis=1)
    if debug:
        plt.plot(center1[1], center1[0], 'o', markersize=4)
        plt.plot(center2[1], center2[0], 'o', markersize=4)
        plt.close()
        
    return center1, center2

def freq_to_T(f):
    """
    mhz to s
    """
    return 1/(f/1000)

def get_times(T_set, timestostable, timestowatch, fps, freq_specific=False):
    """
    Parameters
    ----------
    T_set : TYPE
        periods set.

    Returns
    -------
    timestamps : 
        frame numbers to analyze
    freqs :
        frequencys to match the timestamps
    """
    start = 0
    timestamps = []
    freqs = []
    for T in T_set:
        period      = T*fps
        start       = start + timestostable*period
        time_start  = int(start+period*timestowatch)
        time        = int(period*timestowatch)
        stamps = list(
            np.linspace(int(start), time_start, time, dtype=int))
        if freq_specific:
            timestamps.append(stamps)
        else:
            timestamps = timestamps + stamps
        start = start+period*timestowatch
        single_frequency = 1000/T
        freqs.append(np.full_like(stamps, single_frequency))
    return timestamps, freqs

def find_color(img, color, q, show_im=False):
    yellow_matrix = np.sum((img - color)**2, axis=-1)
    indeces_fixed = np.where(yellow_matrix < q)
    if show_im:
        n = len(color)
        plt.figure()
        govno = np.repeat([yellow_matrix.T], n, axis=0).T
        fixed_img = np.where(govno < q,
                             img, np.full_like(img, 7.0))
        plt.imshow(fixed_img)
    return indeces_fixed

def mp_init(shared_data):
    global color1
    global color2
    global video_path

    video_path = shared_data.cap
    color1 = shared_data.color1
    color2 = shared_data.color2
    pass

def mp_single_frequency(arg_array):
    global video_path
    global color1
    global color2
    debug = False
    __coors_left = []
    __coors_right = []
    __freq, __timestamps = arg_array
    cap = cv.VideoCapture(filename=video_path, apiPreference=cv.CAP_FFMPEG)
    for i, frame_number in enumerate(__timestamps):
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number-1)
        ret, frame = cap.read()
        kernel = np.ones((3, 3), np.uint8)
        frame = cv.dilate(frame, kernel, iterations=3)   
        frame = cv.erode(frame, kernel, iterations=3)   
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if debug:
            plt.figure()
            plt.imshow(frame)
            plt.close()
        __coor1, __coor2 = get_coors(frame, color1, color2, debug)
        __coors_left.append(__coor1)
        __coors_right.append(__coor2)
        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    return (__freq, __coors_left, __coors_right)

def mp_analyze_video(video_path, timestamps, frequencys,
                     color1, color2,
                     progress_bar=True):

    def amp(data):
        Y = np.asarray(data)[:, 0]
        return np.abs(np.max(Y) - np.min(Y))

    coors_left = []
    coors_right = []
    shared_data = shared_data_class(video_path, color1, color2)
    pool = mp.Pool(processes=mp.cpu_count(),
                   initializer=mp_init, initargs=(shared_data, ))
    N_t = len(frequencys)
    amplitudes_left = []
    amplitudes_right = []
    result_freqs = []
    freqs_full = []
    if progress_bar:
        from tqdm import tqdm

        with tqdm(total=N_t, maxinterval=0.1) as pbar:
            for res in pool.imap(mp_single_frequency, zip(frequencys, timestamps)):
                result_freq = res[0]
                freqs_full.extend(res[0])
                coors_left.extend(res[1])
                coors_right.extend(res[2])
                np.savetxt(
                    f'data/left_{result_freq[0]}.txt', res[1])
                np.savetxt(
                    f'data/right_{result_freq[0]}.txt', res[2])
                amplitudes_left.append(amp(res[1]))
                amplitudes_right.append(amp(res[2]))
                result_freqs.append(result_freq[0])
                np.savetxt('data/amps.txt', np.vstack([amplitudes_left,
                                                       amplitudes_right,
                                                       result_freqs]).T)
                pbar.update(1)
            pool.close()
            pool.join()
            pbar.update()
            time.sleep(2.0)
    return freqs_full, coors_left, coors_right


limit = [200, 3000]  # в мгц
TimesToStable = 8.8
TimesToWatch = 1.2
start, end = limit
step = int(2)  # в мгц
fps = 60
path1 = f'{start}-{end}.MOV'
if __name__ == "__main__":

    freq_set = np.linspace(start, end, (end-start)//step+1, dtype=int)
    t_set = list(map(freq_to_T, freq_set))

    time, freqs = get_times(t_set, TimesToStable,
                            TimesToWatch, fps=fps, freq_specific=True)
    color1 = [90, 225, 225]
    color2 = [55, 175, 125]
    print(cv.getBuildInformation())
    freq_full, coor1, coor2 = mp_analyze_video(
        path1, 
        time,
        freqs, 
        color1, 
        color2)
    np.savetxt(f'data/full_left{start}-{end}.txt', coor1)
    np.savetxt(f'data/full_right{start}-{end}.txt', coor2)
    np.savetxt(f'data/full_freq{start}-{end}.txt', freq_full)

    print(f'{start}-{end} ended')
