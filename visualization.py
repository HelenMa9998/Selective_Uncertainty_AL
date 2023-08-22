import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import pandas as pd

plt.style.use("fivethirtyeight")
plt.figure(figsize=(10, 7))
# plt.figure(dpi=100,figsize=(5,3)) # 分辨率参数-dpi，画布大小参数-figsize
data = pd.read_csv('./messidor/baseline/performance.csv')

xdata = data.iloc[:, data.columns.get_loc('Training dataset size')]
ydata = data.iloc[:, data.columns.get_loc('Accuracy')]

plt.plot(xdata,ydata,'cornflowerblue',label='performance',linewidth=1)

# orangered
my_x_ticks = np.arange(100, 768, 100)
plt.xticks(my_x_ticks)
my_y_ticks = np.arange(0.64, 0.9, 0.04)
plt.yticks(my_y_ticks)

plt.title("Messidor",size=15)

plt.xlabel('Training dataset size',size=12)
plt.ylabel('Testing Accuracy',size=12)
plt.axhline(y=0.8667, color = 'grey', linewidth=1.5, linestyle="-", label='Inception v3' )
plt.legend(frameon=True,borderaxespad=0, prop={'size': 15, "family": "Times New Roman"})
plt.savefig('./deepfool.jpg', format='jpg',  bbox_inches='tight', transparent=True, dpi=600)