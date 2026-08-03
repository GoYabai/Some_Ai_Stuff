import numpy as np

sample = []
N, M = map(int, input().split())
for i in range(N):
    sample.append(list(map(int, input().split())))
sample = np.array(sample)
print(np.mean(sample, axis = 1))
print(np.var(sample, axis = 0))
print(round(np.std(sample), 11))
