#运动分类实例编写
流程 1.从特征文件标签文件中将所有数据加载到内存中
       由于有缺失值，需要简单的预处理

     2.根据算法创建对应的分类器，并使用训练数据进行训练

     3.利用测试集进行预测，评价模型



import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer　　　　　　#导入预处理模块
from sklearn.model_selection import train_test_split  #自动生成训练集合测试集的模块
from sklearn.metrics import classification_report     #预测结果评估函数

导入三个分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklean.native_bayes import GaussianNB      #高斯朴素贝叶斯


#数据导入函数
def load_dataset(feature_paths,label_paths):
    #读取特征文件列表和标签文件列表中的内容，归并后返回

    feature= np.ndarray(shape=(0,41))
    label = np.ndarray(shape=(0,1))


#读取数据
    for file in feature_paths:
        #逗号分隔符读取数据，问号替换标记为缺失值，文件不含表头
        df = pd.read_table(file,delimiter=',',na_values='?',header=None)
        #用平均值补全缺失值
        imp = Imputer(missing_values='NAN',strategy='mean',axis=0)
        imp.fit(df)      #fit用于训练预处理器
        df = imp.transform(df) #transform生成预处理结果
        #将新读入的数据合并到特征集合中
        feature = np.concatenate((feature,df)) #将预处理后的结果加入feature，依次遍历完所有文件
        
#标签文件处理函数
    for file in label_paths:
        #读取标签数据，不含表头
        df = pd.read_table(file,header=None)
        #新读入的数据合并到标签集合中
        label = np.concatenatenate((label,df))

    #标签归整为一维向量
    label = np.ravel(label)
    return feature,label   #返回特征值与标签值


#主函数数据准备
if __name__=='__main__':
    #设置数据路径
    feature_paths = [A,B,C,D,E]
    label_paths = [A1,B2,C3,D4,E5]

    #前四个数据作为训练集导入
    x_train,y_traiin = load_dataset(feature_path[:4],label[:4])
    #最后的数据作为测试集导入
    x_test,y_test = load_dataset(feature_path[4:],label_paths[4:])


    #使用全量数据作为训练集，借助train_test_split函数将训练数据打乱
    x_train,x_,y_train,y_ = train_test_split(x_train,y_train,test_size=0.0) #测试集比列为0，将数据随机打乱

    #knn
    knn = KNeighborsClassifier().fit(x_train,y_train)
    answer_knn = knn.predict(x_test)    #测试集进行分类器预测，得到分类结果


    #tree
    dt = DecisionTreeClassifier().fit(x_train,y_train)
    answer_dt = dt.predict(x_test)

    #bayes
    gnb = GaussianNB().fit(x_train,y_train)
    answer_gnb = gnb.predict(x_test)
    


#计算准确率和召回率
    classification_report(y_test,answer_knn)

    classification_report(y_test,answer_dt)

    classification_report(y_test,answer_gnb)
    




        
