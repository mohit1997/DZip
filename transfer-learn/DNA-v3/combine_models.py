import numpy as np


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)

l1 = np.load('original_prob.npy')
l2 = np.load('multitransfer_explosslist.npy')

print(len(l1), len(l2))
window = 2

print(np.mean(l1, dtype=np.float64))
print(np.mean(l2, dtype=np.float64))

loss1 = np.mean(strided_app(l1, window, window), axis=1, dtype=np.float64)
loss2 = np.mean(strided_app(l2, window, window), axis=1, dtype=np.float64)

final_l = [loss1[0]]
for i in range(1, len(loss2)):
    if loss1[i-1] < loss2[i-1]:
        final_l.append(loss1[i])
    else:
        final_l.append(loss2[i])

print(np.mean(final_l))
