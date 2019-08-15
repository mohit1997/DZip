import numpy as np


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n), writeable=False)

l1 = np.load('transfer_explosslist.npy')
l2 = np.load('loss2.npy')
print(l1.shape, l2.shape)
window =52

loss1 = np.mean(strided_app(l1, window, window), axis=1)
loss2 = np.mean(strided_app(l2, window, window), axis=1)

final_l = [loss1[0]]
for i in range(1, len(loss2)):
    if loss1[i-1] < loss2[i-1]:
        final_l.append(loss1[i])
    else:
        final_l.append(loss2[i])

print(np.mean(final_l))
