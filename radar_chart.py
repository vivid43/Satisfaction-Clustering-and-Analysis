import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def draw_radarchart(i,j,n,y_min,y_max,values,feature,title,outfile):
    '''
    :param i: i items a row
    :param j: j items a column
    :param values: hospital score rank
    :param feature: 1st level index name
    :param title: hospital name
    :param outfile: picture save_path
    :return: None
    '''
    # set picture style,fonts, ,size, subplot
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams["figure.figsize"] = (i*4-1, j*4-1)
    gs = plt.GridSpec(j, i)
    fig = plt.figure()
    for k in range(n):
        value = values[k]
        N = len(value)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        value = np.concatenate((value, [value[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        ax = fig.add_subplot(gs[int((k) / i), k % i], polar=True)

        ax.plot(angles, value, 'o-', linewidth=0.5)
        ax.fill(angles, value, alpha=0.25)
        ax.set_thetagrids(angles * 180 / np.pi, feature)

        ax.set_ylim(y_min, y_max)
        plt.title(title[k])
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=100)


if __name__ =='__main__':
    rank = pd.read_csv('data/rank1.csv')
    rank.sort_values(by='All', ascending=False, inplace=True)

    values = rank.iloc[:,1:7].values
    feature = rank.columns[1:7]
    title = rank['O_NAME'].values

    draw_radarchart(4, 6, 24,0,43, values[:24], feature, title[:24], 'pic/24.png')
    draw_radarchart(4, 5, 20,0,43, values[24:44], feature, title[24:44],'pic/20.png')
