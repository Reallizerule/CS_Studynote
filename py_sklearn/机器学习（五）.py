#有监督学习
    利用一组带有标签的数据，学习从输入到输出的映射，然后将这种
    映射关系应用到未知的数据上，达到分类或回归的目的

    1.输出为离散的，学习任务为分类任务     
    2.输出是连续的，学习任务为回归任务



    #分类学习
      输入：一组有标签的训练数据（也叫观测和评估），标签表明数据的所属类别

      输出：分类模型根据训练的数据，训练自己的模型参数，学习出一个
            适合这组数据的分类器，有新数据时将数据输入给学习好的分类器
            进行判断。


      #分类学习中的训练集与测试集
        训练集：用来训练模型的已标注好的数据，是建立模型，发现规律用的

        测试集：还是以标注的数据。通常做法是隐藏标注，输送给训练好的模型
                通过结果与真实标注比较，评估模型的学习能力。


         #一个简单划分测试集训练集的方法是随机从已标注数据抽取一部分
         #作为训练数据，其它是测试数据，此外还有交叉验证法
         #自助法评估分类模型


    #分类器评价指标

    精确率：针对预测结果而言，（二分类为例）,表示预测为正的样本中
            有多少是真正的样本，预测为正有两种可能，正类分成正类（TP），负类预测成正类（FP）
            P=TP/TP+FP

    召回率：针对原来的样本而言，表示样本中有多少正类被预测正确了，两种可能。原来的正类预测
            为负类（FN），正类预测成正类（TP）
            P=TP/FN+TP

    准确率：预测对的/所有
            =（TP+TN）/(TP+FN+FP+TN)



    #sklearn 中聚类算法都封装在sklearn.cluster子模块中
    #而分类的算法并未统一封装，因此import方式各有不同


    #sklearn提供的分类器
        k近邻方法
        朴素贝叶斯
        支持向量机
        决策树
        神经网络
        （包括了线性分类器，也有非线性分类器）
        #线性分类器是依靠线性的空间划分进行分类



    #回归
        回归：统计学分析数据方法，了解两个变数之间是否相关
              研究相关方向与强度，并建立了数学模型以便观察
              特定的变数来预测研究者感兴趣的变数
        特点：帮助人们了解自变量变化时，因变量的变化量
              一般来说可以用回归分析给出自变量估计因变量的
              条件期望。

    #sklearn的回归函数封装在skleran.linear_model（线性回归模型）
                           #sklearn.preprocessing两个子模块里面。

    线性回归函数
         普通回归函数
         岭回归
         lasso


    非线性回归函数
         多项式回归


    #应用：经常应用在时间序列数据分析上，进行预测和数据的拟合。


              
        
#监督学习之分类
    人体运动状态预测-实例分析
    #可穿戴设别让我们可以获取人体各项数据。采集到数据后进行
    #分析和建模，通过各项特征数值进行用户状态的精准判断。
    1.数据介绍
        收集了A,B,C,D,E五个用户的传感器数据，包含特征文件和标签文件
        特征文件：每一行对应一个时刻所有传感器数值
        标签文件：与特征文件对应，记录了该时刻用户的姿态。行数相同
                  相同行直接互相对应。



    2.特征文件包含了41列特征，包括心率，时间戳，磁场强度等。
      标签文件中对应特征文件的行，有25种身体姿态。包括无活动，坐，跑等
      作为训练集的标准参考准则，可以进行特征的监督学习。

    3.假设有一个新用户，但只有传感器采集的数据，如何得到用户的新姿态

    #按图索骥，寻找每一个类别对应的数据规律，继而找到姿态和特征的相关关系。

    在明确是一个分类问题情况下，通过选定某种分类模型，训练数据训练模型学习
    然后对每个测试样本给出对应的分类结果。

    #KNN分类器
    KNN：
        原理：通过计算待分类数据点，与已有数据集中的所有数据点的距离
              取距离最小的前K个点，根据少数服从多数的原则，将数据点划
              分为出现次数最多的类别
        sklearn中使用sklearn.neighbors.KNeighborsClassifier
        创建一个K近邻分类器。主要参数
        n_neighbors:指定分类器中K的大小（默认值为五，注意和kmeans的区别）
        weights:设置K个点对分类结果影响的权重（默认是平均权重uniform）
        可以选择'distance'代表越近的点权重越高，或者传入自己编写的权重
        计算函数。

        #algorithm ：设置计算临近点的方法。因为数据量很大时
                     计算当前点和所有点的距离很耗时间
                     所以有ball_tree，kd_tree,brute分别代表了不同的寻找
                     邻居的优化算法，默认值是auto，根据数据自动选择


    #实例编写
        #创建一组数据X和它对应的标签y
        x=[[0].[1],[2],[3]]
        y=[0,0,1,1]
        #import语句导入K近邻分类器
        from sklearn.neighbors import KNeighborsClassifier
        #设置参数n_neighbors设置为3，即使用最近3个邻居作为分类依据，其它参数默认
        neigh = KNeighborsClassifier(n_neighbors=3)
        #调用fit()函数，将训练数据X和标签y送入分类器进行学习
        neigh.fit(x,y)

        现在分类器就可以根据我们的训练数据学习好了
        #K近邻分类器的使用
        调用predict()函数，对未知的样本[1.1]分类，并且直接将
        要分类的的数据构造作为数组形式参数传入，得到分类标签作为返回值
        neigh.predict([[1.1]])
        #这里的输出值是0，表示分类器通过计算距离，取0,1,2这三个邻居
        #根据投票法最终分类为类别0

        实际使用的时候，我们可以使用所有训练数据，构成特征X和标签Y
        使用fit()函数进行训练，在正式分类的时候，通过一次性构造测试集
        或者一个一个输入样本的方式得到样本对应分类结果。

        #如果K值取值较大，相当于用较大邻域中的训练实例进行预测
        #可以减小估计的误差，但是距离较远的样本也会对预测起作用，导致预测错误

        #如果K较小，则用较小的邻域进行预测，如果邻居是噪声点，会导致过拟合。

        #一般取比较小的K值，然后使用交叉验证的方法来选取最优的K值


    #决策树算法
        决策树：决策树是一种树形结构分类器，通过顺序询问分类点的属性
                 决定分类点最终的类别。通常根据特征的信息增益或其他指标
                 构建一颗决策树，在分类的时候按照决策树的节点依次进行判断
                 即可得到样本所属的类别。

        #例子：关于各个属性和信用卡偿还能力的决策树

        sklearn:使用sklearn.tree.DecisionTreeClassifier
                创建一个决策树进行分类，主要参数有

                criterion:用于选择属性的准则，可以传入'gini'代表基尼系数
                          或者'entropy'代表信息增益

                max_features:表示决策树节点进行分裂时，从多少个特征中选择最优
                             的特征，可以设定固定数目，百分比，或者其他标准
                             默认值是使用所有特征个数


    #决策树的实例
        #导入鸢尾花数据集
        from sklearn.datasets import load_iris
        #导入决策树分类器和交叉验证值得函数cross_val_score
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_score
        #使用默认参数，创建一颗基于基尼系数的决策树，将实例赋值给clf
        clf=DecisionTreeClassifier()
        #将鸢尾花数据赋值给变量iris
        iris = load_iris()

        #决策树的使用
        1.我们将决策树分类器作为待评估模型，iris.data鸢尾花数据作为特征
        iris.target鸢尾花分类标签作为目标结果，设定cv（交叉验证）为10
        使用10折交叉验证，得到最终的交叉验证得分。
        cross_val_score(clf,iris,data,iris.target,cv=10)

        2.我们也可以仿造以前K近邻分类器的方法，利用fit函数训练模型，并用
          predict()函数进行预测
          clf.fit(x,y)
          clf.predict(x)
        #决策树本质是寻找特征空间上划分，旨在构建一个训练数据拟合的好
        #并且复杂度小的决策树

        实际使用中，需要根据数据情况，调整DecisionTreeClassifier类中传入的参数
        类中传入的参数，比如选择合适的criterion,设置随机变量等


    #朴素贝叶斯分类
        朴素贝叶斯：以贝叶斯定理为基础的多分类分类器，对于给定的数据，首先基于
                    特征的条件独立假设，学习输入输出的联合概率分布，然后基于此模型
                    对给定的输入x，利用贝叶斯定理求出后验概率最大的输出

        sklearn中三个朴素贝叶斯分类器
                     native_bayes.GussianNB   高斯朴素贝叶斯
                     native_bayes.GussianNB   针对多项式模型的朴素贝叶斯分类器
                     native_bayes.GussianNB   针对多元伯努利模型的朴素贝叶斯分类器


        sklearn中的朴素贝叶斯
                    区别在于假设某一特征的所有熟语某个类别的观测值符合特定的分布
                    如，分类问题包括人的身高，身高符合高斯分布，这类问题适合高斯朴素贝叶斯

                    #sklearn中 使用sklearn.native_bayes.GussianNB创建一个高斯朴素贝叶斯分类器
                    参数有
                    priors : 给定各个类别先验概率，如果是空则按训练数据实际情况统计
                             如果给定先验概率，则在训练过程中不能更改

                    #构造训练数据x,y
                    x=np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
                    y=np.array([1,1,1,2,2,2])
                    #导入朴素贝叶斯分类器
                    from sklearn.native_bayes import GaussianNB
                    #使用默认参数，创建一个高斯朴素贝叶斯分类器，实例赋值给变量clf
                    clf=GaussianNB(priors=None)
                    #类似的使用fit()函数进行训练，并使用predict()函数进行预测。得到预测结果为1
                      （可以测试时构造二维数组达到同时预测多个样本的目的）
                    clf.fit(x,y)
                    print(clf.predict([[-0.8,-1]]))
                    [1]
                    #朴素贝叶斯是典型的生成学习方法由训练数据学习联合概率分布，并且求得后验概率分布
                    #小规模数据上朴素贝叶斯模型表现很好，适合多分类任务






#人体运动状态预测程序编写
                    算法流程：  1.数据预处理，从特征文件和标签文件中将所有数据加载内存中
                                  由于有缺失值，此步骤需要简单数据预处理

                                2.创建对应的分类器，并使用训练数据进行训练

                                3.利用测试集进行测试，使用真实值与预测值对比，计算模型整体的
                                  准确率和召回率，来评测模型。



                    #模块的导入,Imputer为预处理函数，train_test_split自动生成测试集训练集的函数
                    #导入预测结果评估模块classification_report
                                  
                    import numpy as np
                    import pandas as pd

                    from sklearn.preprocessing import Imputer
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import classification_report

                    from sklearn.neighbors import KNeighborsClassifier
                    from sklearn.tree import DecisionTreeClassifier
                    from sklearn.native_bayes import GaussianNB

                    #数据导入函数
                    def load_datast(feature_path,label_paths):
                        #读取特征文件列表与标签文件列表中的内容，归并返回
                    #编写数据导入函数，传入特征文件列表和标签文件列表

                    #定义feature数组变量，列数量与特征维度一致
                    #定义空的标签变量，列数量与标签维度一致
                        feature=np.ndarray(shape=(0,41))
                        label = np.ndarray(shape=(0,1))
                    #数据导入函数 使用pandas的read_table函数读取特征文件，指定分隔符为逗号，缺失值为？号，不含表头
                        for file in feature_paths:
                            #逗号分隔符读取特征数据，问号标记为缺失值
                            df = pd.read_table(file,delimiter=',',na_values='?',header=None)
                            #使用平均值补全缺失值
                            imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
                            imp.fit(df)
                            df = imp.transform(df)
                            #将读取的数据合并到特征集合
                            feature = np.concatenate((feature,df))
                            
                    #Imputer函数通过设定strategy参数为mean，使用平均值补齐缺失数据，fit()函数训练预处理器，transform函数生成预处理结果
                    #预处理后的数据加入feature，依次遍历完所有的文件，得到特征集合

                    for file in label_path:
                        #读取标签数据，文件不含表头
                        df = pd.read_table(file,header=None)
                        #新读取的数据加入到标签集合中
                        label =np.concatenate((label,df))
                    #将标签规整为一维向量
                    label =np.ravel(label)
                    return feature , label


                    主函数中的数据准备部分
                    if __name__='__main__':
                        #设置数据的路径
                        feature_paths = ['A/A.feature','B/B.feature','C/C.feature','D/D.feature','E/E.feature']
                        label_paths = ['A/A.label','B/B.label','C/C.label','D/D.label','E/E.label']
                        #前四个数据作为训练集导入
                        x_train,y_train = load_datasets(feature_paths[:4],label_paths[:4])
                        #最后一个数据作为测试集导入
                        x_test,y_test = load_datasets(feature_paths[4:],label_paths[4:])
                        #使用全量数据作为训练集，借助train_test_split函数将训练数据打乱
                        x_train,x_,y_train,y_=train_test_split(x_train,y_train,test_size=0.0)
                        #使用python切片，将数据路径前四个值作为训练集，并作为参数传入load_datasets函数，得到训练集合特征与标签

                        #最后一个值作为测试集，得到测试集的标签和值

                        由于初始数据按时序储存，我们通过打乱函数，设置测试集比例为test_size=0
                        将数据随机打乱，以便后续的分类器初始化和训练。

                    #k近邻分类器使用，并在测试集上预测.训练后的分类器保存在变量knn中
                        
                        knn = KNeighborsClassifier().fit(x_train,y_train)
                        
                        answer_knn = knn.predict(x_test)


                    #决策树训练，默认参数创建决策树分类器dt，训练后的模型存入dt
                        dt = DecisionTreeClassifier().fit(x_train,y_train)

                        answer_dt = dt.predict(x_test)
                        

                    #朴素贝叶斯分类器
                        gnb = GaussianNB().fit(x_train,y_train)

                        answer_gnb = gnb.predict(x_test)


                    #使用classification_report函数对分类结果精确率，召回率，F1，支持度衡量

                        classification_report(y_test,answer_knn)

                        classification_report(y_test,answer_dt)

                        classfificaton_report(y_test,answer_gnb)
