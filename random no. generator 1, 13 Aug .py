"""""Shivam Kumar 2311166"""""

# Random Number Generator
# my own values of c are 4, 3.78 , 2.89, 3.84, 3.6

from all_lib import random

l = random()
print(l)
xi = l[:-5]
xi_5 = l[5:]

import matplotlib.pyplot as plt


def Plot(x, y, title='Sample Plot', xlabel='X-axis Label', ylabel='Y-axis Label', file_name='sample_plot.png'):
    plt.scatter(x, y, marker='.', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(file_name)
    plt.show()


indices = list(range(len(l)))

# Call plot
Plot(indices, l, title='Random Number Generator Output, c=2.89', xlabel='Xi', ylabel='Xi+5')
