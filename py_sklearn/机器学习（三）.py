#无监督学习中的K-means算法
   以K为参数，将N个对象分为K个簇，使簇内具有较高的相似度，而簇间相似度低
   1.随机选取K个点作为初始的聚类中心
   2.对剩下的点，根据其与聚类中心的距离，将其归入最近的簇
   3.对每个簇，计算所有点的均值，作为新的聚类中心
   4.重复2，3直到聚类中心不在改变


   应用：由全国31个省份居民家庭平均每人消费支出的八个主要变量数据
          包括衣着，医疗保健，交通等，对31个省份进行聚类
          
   实现过程：
   import umtpy
   from sklearn.cluster import KMeans
       #从txt中加载数据
   data,cityName = loadData(city.txt)
   km = KMeans(n_cluster=3)#创建实例
   label = km.fit_predict(data)#调用聚类的方法进行计算,传入序列，输出每个序列对应的标签
   #n_cluster指定聚类中心个数
   #init:初始化聚类中心的初始化方法
   #max_iter：最大迭代次数
   #一般只用给出n-cluster,init默认是kmeans++,max_iter默认300
   #data是加载的数据，label聚类后数据所属标签，fit_predict计算簇中心分配序号
      def loadData(filepath):
          fr =open(filepath,'r+')
          lines = fr.readlines()
          retData=[]
          retCityName=[]
          for line in lines:
              item = line.strip().split(',')
              retCityName.append(item[0])
              retData.append([float(items[i]))
          for i in range(1,len(items)):
              return retData,retCityName

   
   expenses = np.sum(km.cluster_centers_,axis=1)
   print(expenses)
   #将每个城市按label分成设定的簇，将每个簇的城市输出，将每个簇平均花费输出
   #即我们按消费水平n_cluster类，消费水平相近城市聚一个类中
   #expense:聚类中心的数值加和，即平均消费水平
                             
   CityCluster=[[],[],[]]
   for i in range(len(cityName)):
       CityCluster[label[i]].append(cityName[i])#是添加在内置序列里，里面三个序列对应三个标签
   for i in range(len(CityCluster)):
       print("Expenses:%.2f"%expenses[i])
       print(CityCluster[i])
   #拓展与改进
    计算两条数据相似性时，默认使用的是欧氏距离，虽然有多种距离测量方法
    但sklearn并没有对应的参数
    如果需要改进，可以在源代码中改进kmeans



#无监督学习只DBSCAN密度聚类
   是一种基于密度的聚类算法
    1.聚类时不需要预先指定簇的个数
    2.最终簇的个数也不确定
    #算法将数据点分为三类
         1.核心点：半径Eps内含有超过MinPts数目的点（要加上自己）
         2.边界点：半径Eps内点的数量小于minpts,但是落在核心点领域内的点
         3.噪音点：既不是核心点，也不是边界点
    #算法流程：
        1.将所有点标记为核心点，边界点，或噪声点
        2.删除噪声点
        3.为距离在Eps之内的核心点间赋予一条边
        3.每组联通的核心点形成一个簇

    #实例：指定Eps=3,minEps=3，用曼哈顿距离聚类

    #分析学生上网时间与上网时长的模式}
                            
        实验过程：建立工程引入相关的包 加载数据，预处理数据 上网时长与上网时间的聚类分析 分析结果
        主要参数
                eps:两个样本别看做邻居节点的最大距离
                min_samples:簇的样本数
                metric：距离计算方式
                例：sklearn.cluster.DBSCAN(eps=0.5,min_samples=5,metric='euclidean').邻居节点为0.5,样本簇是5，欧氏距离计算

                #上网时间的聚类
                x=real_x[:,0:1]
                db=skc.DBSCAN(eps=0.01,min_samples=20).fit(x)#调用dbscan方法训练，labels为每个簇的标签
                labels=db.labels

                raito = len(labels[labels[:]==-1])/len(labels)#打印数据被记上的标签，计算标签为-1，即噪声数据的比例
                                                              #注意这里是labels数组处理，序列是不行的
                n_cluster = len(set(labels)) - 1 if -1 in labels else 0 #计算聚类后簇的个数，后面用了pythonic的写法
                for i in range(n_cluster):
                             print(i)
                             print(list(x[labels==i].flatten)) #打印各簇的标签，评价聚类的结果
                             metrics.silhouette_score(x,labels)
                pet plt.hist(x,24)#画出直方图，来更直观判断



             #机器学习在聚类分析时，可以采用取对数的方法，让数据变得容易聚类分析


                            
                              
        
   



                             
