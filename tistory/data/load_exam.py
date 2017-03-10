import numpy as np

xy = np.loadtxt('train.txt', delimiter = ' ', unpack=True, dtype='float32'  )
x_data = xy[0:-1]
y_data = xy[-1]

print (x_data, y_data)

