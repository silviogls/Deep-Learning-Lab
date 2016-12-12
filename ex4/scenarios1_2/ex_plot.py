import matplotlib.pyplot as plt
import csv
import numpy as np
import exercise4


### scenario 1

exercise4.scenario(1, 0.0)

f = open('fprofile.txt', 'r')
r = csv.reader(f, delimiter = "\t")
plt.figure()
for line, epochs in zip(r, [1, 3, 9, 27, 81]):
    x = np.arange(0,1,1.0/len(line))
    plt.plot(x, line, label="n epochs = "+str(epochs))
plt.grid(True)
plt.legend()
plt.show()
f.close()


### scenario 2

exercise4.scenario(2, 0.0)

f = open('fprofile.txt', 'r')
r = csv.reader(f, delimiter = "\t")
plt.figure()
for line, epochs in zip(r, [1, 3, 9, 27, 81]):
    x = np.arange(0,1,1.0/len(line))
    plt.plot(x, line, label="n epochs = "+str(epochs))
plt.grid(True)
plt.legend()
plt.show()
f.close()
