"""import dat
import multiprocessing as mp
import Model as m
import random
import time
from ctypes import c_bool, c_char_p


if __name__ == '__main__':
    device = "cuda"
    NUM_ENVS = 8
    mp.set_start_method('spawn', force=True)
    net = m.DDQN(126, 3, 0.1, num_envs=NUM_ENVS).to(device)
    net.share_memory()
    i = mp.Value('i', 1)  # to control current state of the process
                        # 2-reading from queue,
                        # -1 - end of the process
                        # 1-writing to process
    name = mp.Manager().Value(c_char_p, "0")  # contsains the name of thee current environment
    change = mp.Manager().Value(c_bool, False)  # contains the value for changing to new env
    j = mp.Value('i', 0)
    lock = mp.Lock()
    train_queue = mp.Queue(maxsize=3)
    data_proc = mp.Process(target=dat.data_func, args=(net, device, train_queue, i, name, change, j))
    data_proc.start()
    ii = 0
    while True:
        if ii == 3:
            with lock:
                i.value = -1
        time.sleep(0.005)
        with lock:
            if (i.value == 2):
                train_entry = train_queue.get()
                print(train_entry)  # for testing
                ii += 1
            if (i.value == -1):
                break
    print("p")
    data_proc.join()"""
import multiprocessing as mp
import dat
import Model as m
import random
import time
from ctypes import c_bool, c_char_p



if __name__ == '__main__':
    device = "cuda"
    NUM_ENVS = 24
    mp.set_start_method('spawn', force=True)
    net = m.DDQN(126, 3, 0.1, num_envs=NUM_ENVS).to(device)
    net.share_memory()
    i = mp.Manager().Value('i', 1)  # to control current state of the process
                        # 2-reading from queue,
                        # -1 - end of the process
                        # 1-writing to process
    name = mp.Manager().Value(c_char_p, "0")  # contsains the name of thee current environment
    change = mp.Manager().Value(c_bool, False)  # contains the value for changing to new env
    j = mp.Value('i', 0)
    lock = mp.Lock()
    train_queue = mp.Queue(maxsize=3)
    data_proc = mp.Process(target=dat.data_func, args=(net, device, train_queue, i, name, change, j))
    data_proc.start()
    print("1")
    ii = 0
    while True:
        if ii == 3:
            with lock:
                i.value = -1
        time.sleep(0.01)
        with lock:
            if not (i.value==1):
              print(i.value)
            if (i.value == 2):
                print("here")
                train_entry = train_queue.get()
                print(train_entry)  # for testing
                ii += 1
            if (i.value == -1):
                break
    print("p")
    data_proc.join()
