import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from radar_chart import draw_radarchart
import os
from collections import Counter
import glob

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#将类属性封装在config类中，对于不同类型的采用不同的配置
class configuration(object):
    def __init__(self,type):
        self.type = type
    def get_subset(self):
        if self.type == 'mz':
            return ['SEX','AGE', 'EDUCATION', 'PAY','挂号体验', '门诊医生沟通', '门诊护士沟通', '门诊环境与标识', '隐私', '门诊医务人员回应']
        elif self.type == 'zy':
            return ['SEX', 'AGE', 'EDUCATION', 'PAY', '住院护士沟通', '住院医务人员回应','住院医生沟通', '住院环境与标识', '出院信息', '疼痛管理', '药物沟通', '饭菜', '对亲友态度' ]
        elif self.type == 'yg':
            return ['SEX', 'EDUCATION', 'WORK_POST','WORK_LEVER', '员工薪酬福利', '员工发展晋升', '员工工作内容与环境', '员工上下级关系', '员工同级关系']
        else:
            raise('Error!')
    def get_dimensions(self):
        if self.type == 'mz':
            return ['挂号体验', '门诊医生沟通', '门诊护士沟通', '门诊环境与标识', '隐私', '门诊医务人员回应']
        elif self.type == 'zy':
            return ['住院护士沟通', '住院医务人员回应','住院医生沟通', '住院环境与标识', '出院信息', '疼痛管理', '药物沟通', '饭菜', '对亲友态度']
        elif self.type == 'yg':
            return ['员工薪酬福利', '员工发展晋升', '员工工作内容与环境', '员工上下级关系', '员工同级关系']
        else:
            raise ('Error!')

    def get_profile(self):
        return list(set(self.get_subset())-set(self.get_dimensions()))

    def get_sequence(self):
        seq = {'SEX':['男','女'],
                'AGE':['20岁以下','20-29岁','30-39岁','40-49岁','50-59岁','60岁以上'],
                'EDUCATION':['初中及以下','高中或中专','本科或大专','研究生'],
                'PAY': ['公费医疗','城镇医疗保险（职工/居民）','新农合','个人自付'],
               'WORK_POST':['医生','护士','技师','药师','管理人员','后勤人员','其他'],
            'WORK_LEVER':['正高','副高','中级','初级','无职称']
               }
        return seq
    def get_K(self):
        K_value = {'mz':4,'zy':3,'yg':3}
        return K_value[self.type]


# 在WORK_YEAR这个属性中，有很多异常值，需要进行数据预处理

# 载入数据的类
class data_load(object):
    def __init__(self, file_name, type):
        self.file_name = file_name
        self.type = type

    @property
    def config(self):
        return configuration(self.type)

    def get_all(self):
        df = pd.read_csv(self.file_name)
        df.dropna(subset=self.config.get_subset(), inplace=True)
        return df

    def get_profile(self):
        profile = self.get_all()[self.config.get_profile()]
        return profile

    def get_data(self):
        data = self.get_all()[self.config.get_dimensions()]
        return data

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def compute_sat(data,labels,dimensions):
    tem = data.copy()
    tem['labels'] = labels
    df =pd.pivot_table(tem, index=['labels'],
                   values= dimensions,
                   aggfunc=np.average)
    df = df[dimensions]
    return df

def cluster_KM(data,k):
    clf = KMeans(n_clusters=k, init='k-means++')  # 默认类别：3
    clf.fit(data, k)
    labels = clf.predict(data)
    return labels

def arrange(df):
    '''
    :param df: 原始的df
    :return: 改变后的df，以及label的映射表
    '''

    df['sum'] = [df.loc[i,:].sum() for i in df.index]
    df.sort_values('sum',ascending= False, inplace = True)
    df.drop(columns = ['sum'],inplace = True)
    label_dict = {}
    for i,j in enumerate(df.index):
        label_dict[j] = i
    return df,label_dict

def map_labels(labels,label_dict):
    for i in labels:
        i = label_dict[i]
    return labels

def draw_radar(df,labels,outfile):
    k,m = df.shape
    count = Counter(labels)
    n = len(labels)
    title = ['第{0}类人群\n{1:.0%}'.format(str(int(i+1)),count[i]/n) for i in np.arange(k)]
    draw_radarchart(k,1,k,0,100,df.values,np.arange(1,m+1),title,outfile)


def percent(labels, profile, sequence):
    '''
    cluster:类标签
    feature:属性特征
    sequence：排列顺序
    feature_name:表示
    '''
    # none local cluster
    # none local feature
    labels = np.array(labels)
    profile = np.array(profile)
    k = len(np.unique(labels))
    res = {}

    for i in np.arange(k):
        one_class = profile[labels == i]
        res[i] = [Counter(one_class)[j] / len(one_class) for j in sequence]
    return res


def draw_pie(k, percentage, sequence, outfile):
    # set picture style,fonts, ,size, subplot
    plt.style.use('ggplot')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams["figure.figsize"] = ((k + 1) ** 2 + 1, 2 * k - 1)
    gs = plt.GridSpec(1, k)
    fig = plt.figure()
    for i in np.arange(k):
        value = percentage[i]
        ax = fig.add_subplot(gs[0, i])
        ax.pie(value, labels=sequence,
               labeldistance=1.1,
               autopct='%3.1f%%',
               shadow=False,
               startangle=90,
               counterclock=False,
               # textprops={'fontsize': 12, 'color': 'b'},
               pctdistance=0.8)
        plt.title('第{}类人群'.format(int(i + 1)), bbox={'facecolor': '0.8', 'pad': 5})
        # ax.grid(True)
    plt.tight_layout()
    plt.savefig('{}.png'.format(outfile), dpi=100)