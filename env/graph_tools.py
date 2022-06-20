import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def lineplot_by_y(y, color='b'):
    '''Show linear equation on the graph.
    Args:
        b (int): first coefficient.
        a (int): first coefficient.
    
    Kwargs:
        color (str): plot's color.
    '''
    sns.set_theme(style='darkgrid')
    sns.color_palette('rainbow')
    data_x = np.arange(0, y.shape[0])
    data_y = y
    data = pd.DataFrame({'time': data_x, 'UPL': data_y})
    sns.lineplot(x='time', y='UPL', data=data, color=color).set_title('UPL by time')
    plt.show()