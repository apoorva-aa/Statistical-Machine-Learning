#Name: Apoorva Arya
#Roll_num: 2020032

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import cauchy
import statistics

#Answer-1

x = np.arange(-10, 10, 0.001)                          #x-axis
  
plt.plot(x, norm.pdf(x, 2, 1), "b")                    #plotting P(x|w1) vs x in Blue - Figure 1

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, norm.pdf(x, 5, 1), "g")                     #plotting P(x|w2) vs x in Green - Figure 2

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(x, norm.pdf(x, 2, 1)/norm.pdf(x, 5, 1), "y")  #plotting P(x|w1)/P(x|w2) vs x in Yellow - Figure 3

#-------------------------------------------------------------------------------------------------------------------------#

#Answer-3

fig2 = plt.figure()
y = (1+(x-5)**2)/(2 + (x-5)**2 + (x-3)**2)             #using the expression obtained on simplification in the scanned copy
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(x, y, "r")                                    #plotting P(w1|x) in Red - Figure 4
                               
plt.show() 
