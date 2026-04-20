import numpy as np


def extract_time(data):
    time = []
    max_seq_len = 0
    for i in range(len(data)):
        seq_len = len(data[i][:, 0])
        max_seq_len = max(max_seq_len, seq_len)
        time.append(seq_len)
    return time, max_seq_len
