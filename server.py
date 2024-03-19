import numpy as np
import scipy as sp
import matplotlib.pyplot as mplpp

import collections as clt
import itertools as itls
import platform
import threading as td
import struct
import socket

if platform.system() == 'Linux':
    HOST = '127.0.0.1'
else:
    HOST = '0.0.0.0'
PORT = 8080

def receive_data(val_queue: clt.deque, td_down: td.Event):
    local = td.local()
    local.socket_buf_size = 2048
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(20)
        s.bind((HOST, PORT))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f'connected by {addr}')
            while True:
                data = conn.recv(local.socket_buf_size)
                if not data:
                    td_down.set()
                    break
                n = len(data) // 4;
                val = struct.unpack(f'!{n}L', data)
                val_queue.extend(val)

def process_data(val_queue: clt.deque, td_down: td.Event):
    fs = 10_000
    res = 0.2
    N = 2 ** int(np.ceil(np.log2(fs / res)))
    f_analysis = np.array(range(0, N // 2)) * fs / (N // 2)
    v_ref = 5.0
    while not td_down.is_set():
        if len(val_queue) > N:
            buf = np.array(map(lambda x: x(), itls.repeat(val_queue.popleft, N)))
            vol = buf / 4096 * v_ref
            Ft = sp.fft.rfft(x=vol, overwrite_x=True)

            Ft_mag = np.abs(Ft)
            Ft_mag_max = max(Ft_mag)
            Ft_norm_mag = 20 * np.log10(Ft_mag / Ft_mag_max)

            mplpp.plot(f_analysis, Ft_norm_mag)
            mplpp.vlines(f_analysis, 0, Ft_norm_mag)
            mplpp.show()

if __name__ == '__main__':
    val_queue = clt.deque()
    td_down = td.Event()

    receive = td.Thread(target=receive_data, args=(val_queue, td_down))
    process = td.Thread(target=process_data, args=(val_queue, td_down))

    receive.start()
    process.start()

    receive.join()
    process.join()
