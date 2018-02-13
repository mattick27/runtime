import numpy as np 
a= np.load('ex3.npy').tolist()
b = a[:225]
c = a[225:]
np.save('ex31.npy',b)
np.save('ex32.npy',c)