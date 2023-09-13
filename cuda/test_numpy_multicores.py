import numpy as np
m = 3072*8
k = 1024*8
n = 2048*8
a = np.random.random_sample((m, k))
b = np.random.random_sample((k, n))
for i in range(20):
    n = np.matmul(a,b)

#cpu around 100, one core
#cpu around 700-1200, multiple cores