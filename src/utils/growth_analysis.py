import numpy as np


def avg_growth(episodes):
    max_len = max(len(e) for e in episodes)
    arr = []
    for e in episodes:
        padded = np.array(
            [np.pad(e, ((0, max_len-len(e)), (0, 0)), mode='edge')])[0]
        arr.append(padded)
    # compute average plant height over all episodes
    data = np.mean(arr, axis=0)
    avg_len = np.mean(data)
    alive = np.mean([(np.count_nonzero(h > 0.5)/len(h)) for h in data])
    return data, avg_len, alive
