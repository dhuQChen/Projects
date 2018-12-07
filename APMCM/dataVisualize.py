import pandas as pd
import matplotlib.pyplot as plt
import os
from _datetime import datetime
import numpy as np


root_path = 'DB'
dirnames = os.listdir(root_path)

for name in dirnames:
    file_path = 'DB/' + name
    plt.figure(figsize=(12, 8))
    data = pd.read_csv(file_path, header=None)
    x = data.loc[:][0]
    # 更改时间格式
    xs = [datetime.strptime(d, '%Y\\%m').date() for d in x]
    y = data.loc[:][1]
    z = 0
    for k in range(2, 11):
        z += data.loc[:][k]
    plt.plot(xs, y, marker='*', c='b', label='Total Recruitees')
    plt.plot(xs, z, marker='o', c='r', label='Provided Post')
    plt.xticks(rotation=45)
    # plt.yticks(np.linspace(min(y)-10, max(y)+10, 20))
    plt.title(name.split('.')[0] + ' Profession Demand Statistics')
    plt.legend()
    plt.savefig('D:/Jupyter/Code/dataVisual/' + name.split('.')[0] + '.png')
    # plt.show()
