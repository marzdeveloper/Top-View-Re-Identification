import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fig1 = plt.figure(1)
plt.title('FTR/TTR')

#plt.plot(x, y, 'r', marker = 'o', label= "Threshold adattiva AVG")
plt.plot(x1, y1 , marker = 'o', label= "Threshold fissa")
plt.plot(x, y,  marker = 'o', label= "Threshold mista + avg std")
plt.plot(x2, y2,  marker = 'o', label= "Threshold mista 1.2x")
#plt.plot(x3, y3,  marker = 'o', label= "Threshold mista 1.2x")
plt.plot(x4, y4,  marker = 'o', label= "Threshold mista 1.4x")

#plt.plot(x1, y1, marker = 'o', label= "Threshold adattiva MIN")
#plt.plot(x2, y2,  marker = 'o', label= "Threshold adattiva MAX")


#plt.xticks(x)
#plt.yticks(np.arange(0, 110, step=10))
plt.xlabel("FTR (%)")
plt.ylabel("TTR (%)")



plt.grid(True)
plt.legend(loc="lower right")

plt.show()