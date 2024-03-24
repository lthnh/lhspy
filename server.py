import numpy as np
import scipy as sp
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

import collections as clt
import itertools as itls
import threading as td
import time
import struct
import socket

HOST = "0.0.0.0"
PORT = 8080


def receive_data(val_queue: clt.deque, td_down: td.Event, conn_ready: td.Event):
    local = td.local()
    local.socket_buf_size = 2048
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(20)
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f"connected by {addr}")
            conn_ready.set()
            while True:
                data = conn.recv(local.socket_buf_size)
                if not data:
                    td_down.set()
                    break
                n = len(data) // 4
                val = struct.unpack(f"!{n}L", data)
                val_queue.extend(val)


def process_data(val_queue: clt.deque, res_queue: clt.deque, td_down: td.Event, conn_ready: td.Event):
    fs = 10_000
    res = 0.2
    N = 2 ** int(np.ceil(np.log2(fs / res)))
    f_analysis = np.array(range(0, N // 2 + 1)) * fs / (N // 2)
    v_ref = 5.0
    last_value = -1
    while not td_down.is_set():
        val_queue_len = len(val_queue)
        if val_queue_len > N:
            buf = np.fromiter(map(lambda x: x(), itls.repeat(val_queue.popleft, N)), dtype=np.uint32)
            vol = buf / 4096 * v_ref
            Ft = sp.fft.rfft(x=vol, overwrite_x=True)

            Ft_mag = np.abs(Ft)
            Ft_mag_max = max(Ft_mag)
            Ft_norm_mag = 20 * np.log10(Ft_mag / Ft_mag_max)
            res_queue.append((f_analysis, Ft_norm_mag))
        else:
            if val_queue_len != last_value and conn_ready.is_set():
                print(f'collected {val_queue_len}/{N} ({float(val_queue_len/N * 100):.02f}%)')
                last_value = val_queue_len
            time.sleep(2)


def display_data(res_queue: clt.deque, td_down: td.Event):
    app = pg.mkQApp('Plotter')

    pg.setConfigOptions(antialias=True)
    p = pg.plot()
    p.setWindowTitle('FFT of signal')
    curve = p.plot()

    def update():
        if td_down.is_set():
            exit
        if len(res_queue) > 0:
            f_analysis, Ft_norm_mag = res_queue.pop()
            curve.setData(f_analysis, Ft_norm_mag)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(50)

    p.show()

    app.exec()


if __name__ == "__main__":
    val_queue = clt.deque()
    res_queue = clt.deque()
    td_down = td.Event()
    conn_ready = td.Event()

    receive = td.Thread(target=receive_data, args=(val_queue, td_down, conn_ready))
    process = td.Thread(target=process_data, args=(val_queue, res_queue, td_down, conn_ready))
    display = td.Thread(target=display_data, args=(res_queue, td_down))
    receive.start()
    process.start()
    display.start()

    receive.join()
    process.join()
    display.join()
