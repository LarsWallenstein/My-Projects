import multiprocessing as mp


def f(kek):
    lock = mp.Lock()
    with lock:
      kek.value = 2


if __name__ == '__main__':
    k = mp.Manager().Value('i',1)
    # f(k)
    p = mp.Process(target=f, args=(k,))
    p.start()
    p.join()

    print(k.value)


