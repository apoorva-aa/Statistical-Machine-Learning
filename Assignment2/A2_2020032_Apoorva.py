#Name: Apoorva Arya
#Roll_num: 2020032

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import scipy.linalg as la
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.stats import multivariate_normal

#Answer-1a

print("Answer 1a - generated")
#class 1
r = bernoulli.rvs(0.5, size=100)           #corresponding to mu1
r1 = bernoulli.rvs(0.8, size=100)         
combined = np.vstack((r, r1))              #made a 2-d array using the samples generated corresponding to mu1 and mu2
#print(combined)

#class 2
r2 = bernoulli.rvs(0.9, size=100)          #corresponding to mu2
r3 = bernoulli.rvs(0.2, size=100)          
combined1 = np.vstack((r2, r3))            #made a 2-d array using the samples generated corresponding to mu1 and mu2
#print(combined1) 

#----------------------------------------------------------------------------------------------------------------------------#

#Answer 1b
print("\nAnswer 1b")
theta1 = sum(r[i] for i in range(50))
print(theta1/50)
theta2 = sum(r1[i] for i in range(50))
print(theta2/50)

thet1 = []
thet2 = []

for i in range(50):
	sum1 = 0
	for j in range(i+1):
		sum1 += r[j]
	thet1.append(sum1/(i+1))

n = np.arange(0, 50, 1)  

plt.plot(n, thet1, color = 'red')

for i in range(50):
	sum2 = 0
	for j in range(i+1):
		sum2 += r1[j]
	thet2.append(sum2/(i+1))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(n, thet2, color = 'red')   

#----------------------------------------------------------------------------------------------------------------------------#

#Answer 1c
print("\nAnswer 1c")
theta1b = sum(r2[i] for i in range(50))
print(theta1b/50)
theta2b = sum(r3[i] for i in range(50))
print(theta2b/50)

thet1b = []
thet2b = []

for i in range(50):
	sum1 = 0
	for j in range(i+1):
		sum1 += r2[j]
	thet1b.append(sum1/(i+1))

n = np.arange(0, 50, 1)  

fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(n, thet1b, color = 'orange')   


for i in range(50):
	sum2 = 0
	for j in range(i+1):
		sum2 += r1[j]
	thet2b.append(sum2/(i+1))

fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
ax3.plot(n, thet2b, color = 'orange')   

#----------------------------------------------------------------------------------------------------------------------------#

#Answer 1d

print("\nAnswer 1d - plotted")
tr_sample = []
tr_sample1 = []
tr_sample2 = []
tr_sample3 = []
n1 = []
for i in range(50):
	tr_sample.append(r[i])
	tr_sample1.append(r1[i])
	tr_sample2.append(r2[i])
	tr_sample3.append(r3[i])
	n1.append(i+1)


fig4 = plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
ax4.scatter(n1, tr_sample, color = 'blue')   

fig5 = plt.figure()
ax5 = fig5.add_subplot(1, 1, 1)
ax5.scatter(n1, tr_sample1, color = 'blue')   

fig6 = plt.figure()
ax6 = fig6.add_subplot(1, 1, 1)
ax6.scatter(n1, tr_sample2, color='magenta')   

fig7 = plt.figure()
ax7 = fig7.add_subplot(1, 1, 1)
ax7.scatter(n1, tr_sample3, color='magenta')   

#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------#

#Answer 3c
print("\nAnswer 3c")
#getting the eigenevalues and eigenvectors
m = np.array([[1, -1],
              [-1, 1]])
pca = PCA()
pca.fit(m)

print("\n---Eigenvalues----")
print(pca.explained_variance_)          #eigenvalues
print("\n---Eigenvectors---")
print(pca.components_)                  #eigenvectors

k = pca.components_                     #U' - Transpose of U
k = np.transpose(k)

print("\n---Y = U'Xc---")
result = np.dot(k, m)                   #U'Xc = Y
print(result)

k = np.transpose(k)
res = np.dot(k, result)                  #UY
og = [2, 4, 5, 3]                        #original matrix elements
  
new = [res[0][0]*(-1)+3, res[0][1]*(-1)+3, res[1][0]*(-1)+4, res[1][1]*(-1)+4]          #new elements
  
# Calculation of Mean Squared Error (MSE)
error = mean_squared_error(og,new)
print("\n---MSE (Mean-Squared-Error)---")
print(error)

#----------------------------------------------------------------------------------------------------------------------------#

#Answer 3d
print("\nAnswer 3d")
d = 5                                              #dimension - modifiable
N = 5                                                #number of samples - modifiable
covar = 0.5                                          #covariance - modifiable
mean = np.zeros(d)                                 #mean - modifiable
X = multivariate_normal.rvs(mean, covar, size = N)   #data matrix

#----------------------------------------------------------------------------------------------------------------------------#

# #Answer 3e
print("\nAnswer 3e")
mu = []
for i in range(d):
    s = 0
    for j in range(N):
        s = s + X[i][j]
    mu.append(s/N)

Xc = []
for i in range(d):
    s = []
    for j in range(N):
        s.append(X[i][j]-mu[i])
    Xc.append(s)

Xc = np.asarray(Xc)
XcT = np.asarray(Xc).T
sig = np.zeros((d,d))
sig = np.dot(Xc,XcT)
sig = sig/d

pca = PCA()
pca.fit(sig)

u = pca.components_
print("Eigen vectors(matrix U): ")
print(u)


#----------------------------------------------------------------------------------------------------------------------------#

#Answer 3f
print("\nAnswer 3f")
uT = u.T
y = np.zeros((d,N))
y = np.dot(uT,Xc)

ans1 = np.dot(u,y)
ans = []
for i in range(d):
    s = []
    for j in range(N):
        s.append(ans1[i][j]+mu[i])
    ans.append(s)
ans = np.asarray(ans) 

mse = mean_squared_error(X.reshape((1,d*N)),ans.reshape((1,d*N)))
print("\n---MSE (Mean-Squared-Error)---")
print(mse)



#----------------------------------------------------------------------------------------------------------------------------#

#Answer 3g
print("\nAnswer 3g - plotted")
mse = []
f = []
for p in range(1,N+1):
    uNew = u[:,:p]
    uTNew = uNew.T
    y = np.zeros((p,N))
    y = np.dot(uTNew,Xc)
    ans1 = np.dot(uNew,y)
    ans = []
    for i in range(d):
        s = []
        for j in range(N):
            s.append(ans1[i][j]+mu[i])
        ans.append(s)
    ans = np.asarray(ans) 
    mse = mean_squared_error(X.reshape((1,d*N)),ans.reshape((1,d*N)))
    f.append(mse)
f2 = np.linspace(1,5,5)  
fig8 = plt.figure()
ax8 = fig8.add_subplot(1, 1, 1) 
ax8.plot(f2,f, color='green')   

plt.show()

#----------------------------------------------------------------------------------------------------------------------------#


