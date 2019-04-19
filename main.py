import glob
import os
import numpy as np
from statis import *

if __name__ == "__main__":
    #files = glob.glob('*.csv')
    files = ['zy-shuxing.csv','yg-shuxing.csv', 'zy-shuxing.csv']
    for file_name in files:
        type = file_name.strip().split('-')[0]
        config = configuration(type=type)
        data_loader = data_load(file_name = file_name,type =  type)

        profile = data_loader.get_profile()
        data = data_loader.get_data()


        folder_name = file_name.strip().split('.')[0]
        make_folder(folder_name)

        k = config.get_K()
        labels = cluster_KM(data,k)

        df = compute_sat(data, labels, config.get_dimensions())
        df, label_dict = arrange(df)
        labels = map_labels(labels, label_dict)


        #将profile，data，label，classes存在一起
        all = np.concatenate([profile.values, data.values, np.array(labels).reshape(-1, 1)], axis=1)
        cols = config.get_subset() + ['labels']
        all = pd.DataFrame(all, index=profile.index, columns=cols)
        all.to_csv('{}/{}_data.csv'.format(folder_name, type))


        # 聚类并画出每类的雷达图
        draw_radar(df, labels, '{}/cluster_radar'.format(folder_name))
        df.to_csv('{}/cluster_feature.csv'.format(folder_name), index=False)

        # 画每个属性在其中分布的饼图
        sequence = config.get_sequence()
        for i in profile.columns:
            percentage = percent(labels, profile[i], sequence[i])
            draw_pie(k, percentage, sequence[i], '{}/{}'.format(folder_name, str(i)))

        print('One File Has Been Done!')