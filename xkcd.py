import matplotlib.pyplot as plt
import numpy as np



plt.xkcd()

x = np.random.uniform(0,1,1000) 
y = np.random.uniform(0,1,1000) 

plt.scatter(x,y)
plt.show()
