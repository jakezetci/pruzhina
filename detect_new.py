# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:34:24 2023

@author: cosbo
"""


import imageio.v2 as iio
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time as timePY
from collections import namedtuple
import pandas as pd
import os

from scipy.optimize import minimize
from progress.bar import IncrementalBar

import multiprocessing as mp

shared_data_class = namedtuple(
    'shared_data_class', ['cap', 'color1', 'color2'])


def get_coors(frame, __color1, __color2, debug=False):
    if debug:

        center1 = find_center(find_color(
            frame, __color1, 5000, True), (400, 700))
        center2 = find_center(find_color(
            frame, __color2, 8000, True), (400, 700))
    else:

        center1 = find_center(find_color(
            frame, __color1, 5000, False), (750, 550))
        center2 = find_center(find_color(
            frame, __color2, 8000, False), (750, 550))
    return center1, center2


def get_stops(freqs):
    stops = np.zeros(len(freqs), dtype=int)
    for i, single_freq in enumerate(freqs):
        stops[i] = int(stops[i-1] + len(single_freq))
    return stops


def freq_to_T(f):
    """
    mhz to s
    """
    return 1/(f/1000)


def read_video(video_path, max_frames=10, skip=5):
    cam = cv.VideoCapture(video_path)
    frames = []
    for i in range(max_frames*skip):
        if i % skip == 0:
            ret, frame = cam.read()

            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(rgb)

            # Display the resulting frame
            if cv.waitKey(1) == ord('q'):
                break

# When everything done, release the capture
    cam.release()
    cv.destroyAllWindows()
    return frames


def get_times(T_set, timestostable=15, timestowatch=3, fps=60, freq_specific=False):
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
        period = T*fps  # period in frames number
        start = start + timestostable*period
        stamps = list(np.linspace(int(start), int(
            start+period*timestowatch), int(period*timestowatch), dtype=int))
        if freq_specific:
            timestamps.append(stamps)
        else:
            timestamps = timestamps + stamps
        start = start+period*timestowatch
        single_frequency = 1000/T
        freqs.append(np.full_like(stamps, single_frequency))
    return timestamps, freqs


def analyze_video_black(video_path, timestamps,
                        color1=[230, 217, 28], color2=[0, 0, 0], max_frames=10e6):
    """
    returns coordinates of time of a single video
    """
    def get_coors(frame, debug=False):
        if debug:

            center1 = find_center(find_color(
                frame, color1, 6000, True), (400, 400))
            center2 = find_center(find_black(
                frame, color2, 10, True), (400, 700))
        else:

            center1 = find_center(find_color(
                frame, color1, 6000, False), (750, 550))
            center2 = find_center(find_black(
                frame, color2, 10, False), (750, 550))
        return center1, center2

    cap = cv.VideoCapture(video_path)
    coors_left = []
    coors_right = []
    debug = False
    for frame_number in timestamps:
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number-1)

        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        coor1, coor2 = get_coors(rgb, debug=debug)
        coors_left.append(coor1)
        coors_right.append(coor2)

        # Display the resulting frame
        if cv.waitKey(1) == ord('q'):
            break

# When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    return coors_left, coors_right


def analyze_video(video_path, timestamps, frequencys,
                  color1=[230, 217, 28], color2=[0, 0, 0], max_frames=10e6):
    """
    returns coordinates of time of a single video
    """

    cap = cv.VideoCapture(video_path)
    coors_left = []
    coors_right = []
    debug = True
    tic = timePY.perf_counter()
    stoppoints = get_stops(freqs)
    bar = IncrementalBar('Progress', max=len(timestamps))

    for i, frame_number in enumerate(timestamps):
        bar.next()
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number-1)

        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Our operations on the frame come here
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        if debug is True:
            plt.imshow(rgb)
        coor1, coor2 = get_coors(rgb, debug=debug)
        coors_left.append(coor1)
        coors_right.append(coor2)
        if i in stoppoints:
            np.savetxt(f'left{video_path[0:-4]}.txt', coors_left)
            np.savetxt(f'right{video_path[0:-4]}.txt', coors_right)
            print('\n')
            print(f'one freq done in {timePY.perf_counter() - tic:.2} s')
            print('\n')
            tic = timePY.perf_counter()
        # Display the resulting frame
        if cv.waitKey(1) == ord('q'):
            break

# When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    return coors_left, coors_right


def find_color(img, color, q=6000, show_im=False):

    yellow_matrix = np.sum((img - color)**2, axis=-1)

    # q = 2*200*200  # порог

    indeces_fixed = np.where(yellow_matrix < q)
    # plt.imshow(yellow_matrix)
    if show_im:
        n = len(color)
        plt.figure()
        govno = np.repeat([yellow_matrix.T], n, axis=0).T  # говнокод
        fixed_img = np.where(govno < q,
                             img, np.full_like(img, 7.0))

        plt.imshow(fixed_img)
    return indeces_fixed


def find_black(img, color=[0, 0, 0], q=100, show_im=False):
    """
    это костыль, который я сделал, чтобы хоть как-то трекать правую метку
    """
    boxed_img = img[:, 400:800, :]  # магия
    black_matrix = np.sum((boxed_img - color)**2, axis=-1)

    indeces = np.where(black_matrix < q)
    indeces_fixed = np.asarray(indeces).T + np.asarray([0, 400])
    if show_im:
        n = len(color)
        plt.figure()
        govno = np.repeat([black_matrix.T], n, axis=0).T  # говнокод
        fixed_img = np.where(govno < q,
                             boxed_img, np.full_like(boxed_img, 7.0))
        plt.imshow(fixed_img)
    return indeces_fixed.T


def find_center(indeces, x0=(170, 170)):
    def all_dist(x0, indeces):

        summ = np.sum((x0 - indeces)**2)
        aa = x0 - indeces
        bb = aa**2
        return summ

    indeces = np.asarray(indeces).T
    result = minimize(all_dist, x0, args=indeces, tol=1e-6)
    center = (int(result.x[0]), int(result.x[1]))

    return center


def find_amplitude(data, freqs):
    A_array = np.zeros(len(freqs))
    freq_array = np.zeros(len(freqs))
    start = 0
    for i, frequency in enumerate(freqs):
        N = len(frequency)
        coors = data[start:start+N]
        Y = coors[:, 0]
        amp = np.abs(np.max(Y) - np.min(Y))
        A_array[i] = amp
        start = start+N
        freq_array[i] = frequency[0]
    return A_array, freq_array


def mp_single_frame(frame_number):
    global cap

    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number-1)

    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        raise ValueError()

    # Our operations on the frame come here
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    debug = False
    if debug is True:
        plt.imshow(rgb)
    coor1, coor2 = get_coors(rgb, debug=debug)

    return (coor1, coor2)


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
    cap = cv.VideoCapture(video_path)
    for i, frame_number in enumerate(__timestamps):
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number-1)
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        if debug is True:
            plt.imshow(rgb)
        __coor1, __coor2 = get_coors(rgb, color1, color2, debug=debug)
        __coors_left.append(__coor1)
        __coors_right.append(__coor2)
        # Display the resulting frame
        if cv.waitKey(1) == ord('q'):
            break
    return (__freq, __coors_left, __coors_right)


def mp_analyze_video(video_path, timestamps, frequencys,
                     color1=[230, 217, 28], color2=[0, 0, 0], max_frames=10e6,
                     progress_bar=True):

    def amp(data):
        Y = np.asarray(data)[:, 0]
        return np.abs(np.max(Y) - np.min(Y))

    def mp_single_frame(frame_number):
        global video_path
        cap = cv.VideoCapture(video_path)
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number-1)

        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            raise ValueError()

        # Our operations on the frame come here
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        debug = False
        if debug is True:
            plt.imshow(rgb)
        coor1, coor2 = get_coors(rgb, debug=debug)
        return (coor1, coor2)

    coors_left = []
    coors_right = []
    debug = False
    tic = timePY.perf_counter()
    stoppoints = get_stops(freqs)
    results = []
    bar = IncrementalBar('Progress', max=len(timestamps))
    shared_data = shared_data_class(video_path, color1, color2)
    pool = mp.Pool(processes=6,
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
            timePY.sleep(2.0)
    return freqs_full, coors_left, coors_right


limit = [50, 5000]  # в мгц

TimesToStable = 12
TimesToWatch = 3
start, end = limit
step = int(5)  # в мгц
fps = 60
path1 = f'{start}-{end}.MOV'

if __name__ == "__main__":

    freq_set = np.linspace(start, end, (end-start)//step+1, dtype=int)
    t_set = list(map(freq_to_T, freq_set))

    time, freqs = get_times(t_set, TimesToStable,
                            TimesToWatch, fps=fps, freq_specific=True)
    time = time[:]
    color1 = [225, 217, 60]
    color2 = [80, 240, 80]
    freq_full, coor1, coor2 = mp_analyze_video(
        path1, time, freqs, color1, color2)
    np.savetxt(f'data/full_left{start}-{end}.txt', coor1)
    np.savetxt(f'data/full_right{start}-{end}.txt', coor2)
    np.savetxt(f'data/full_freq{start}-{end}.txt', freq_full)

    print(f'{start}-{end} ended')
