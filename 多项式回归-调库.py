import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# prepare data
excel = pd.read_excel(r"C:/Users/华为/Desktop/西安疫情.xls")
x = np.array(excel['a']).astype(np.float)
y = np.array(excel['b']).astype(np.float)
    # print(x)
    # print(y)
x = np.array(x)
y = np.array(y)

coef1 = np.polyfit(x,y, 10)
poly_fit1 = np.poly1d(coef1)
plt.plot(x, poly_fit1(x), 'g',label="10th order fitting")
print(poly_fit1)
plt.scatter(x, y, color='red')
plt.legend(loc=2)
plt.show()