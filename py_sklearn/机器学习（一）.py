#sklearn库
  一个基于python的第三方模块，集成了一些常用的机器学习算法。
  我们进行机器学习任务时，不需要实现算法，只需要调用其中的一些模块就可以
  完成任务

  sklearn中的常用数据集 
  波士顿房价数据集  load_boston()   回归  506*13
  鸢尾花数据集      load_iris()     分类  150*4
  糖尿病数据集      load_diabetes() 回归  442*10
  手写数字训练集    load_digits()   分类  5620*24

  脸部图像数据集    fech_olive_face() 降维  400*64*64
  新闻分类数据集    fech_20newsgroups() 分类
  带标签的人脸数据集 fech_lfw_people()  分类，降维
  路透社新闻语料数据集 fech_revl()      分类  804414*47236
  
#波士顿房价数据集
  sklearn.datasets.load_boston
  return_X_y 表示是否返回target，默认为Flase,只返回data（属性）

  from sklearn.datasets import load_boston
  boston = load_boston()
  print(boston.data.shape)

#鸢尾花的数据集
  包括了鸢尾花测量数据与所属类别
  测量数据包括，萼片长度，萼片宽度，花瓣长度，花瓣宽度
  类别分为三类，数据集可用于多分类问题

  return_X_y True,则以（data,target）形式返回数据，默认为False
  表示以字典形式返回数据全部信息（data:target）
  from sklearn.datasets import load_iris
  iris = load_iris()
  list(iris.target_names)

#手写数字数据集
  包括了1797个手写数字数据，每个数字由8*8矩阵构成，矩阵中值范围是0-16
  ，表示颜色的深度，0就是一个8*8的矩阵，里面的值表示颜色深度

  参数与鸢尾花数据集一样
  n_class为特别属性，表示返回数据的类别数，如n_class=5,则返回0到4的数据样本
  from sklearn.datasets import load_digits
  digits = load_digits()

  import matplotlib.pyplot as plt
  plt.matshow(digits.images[0])
  plt.show()



#sklern库基本功能
  分为6大部分，分别用于完成分类任务，回归任务，聚类任务，降维任务
  模型选择与数据预处理

  分类任务：
      最临近算法：neighbors.NearestNeighbors
      支持向量机：svm.SVC
      朴素贝叶斯：native_bayes.GaussianNB
      决策树： tree.DecissionTreeClassifiter
      集成方法: ensemble.BaggingClassifier
      神经网络：neural_network.MLPCLassifier
  回归任务：
      岭回归： linear_model.Ridge
      lasson回归: linear_model.Lasso
      弹性网络: linear_model.ElasticNet
      最小角回归: linear_model.Lars
      贝叶斯回归: linear_model.BayesianRidge
      逻辑回归：  linear_model.LogisticRegression
      多项式回归: preprocessing.PolynomiaFeatures
  聚类任务：
       K-means: cluster.KMeans
       AP聚类:  cluster.AffinityPropagation
       均值飘移： cluster.MeanShift
       层次聚类： cluster.AgglomerativeClustering
       DBSCAN：  cluster.DBSCAN
       BIRCH：   cluster.Birth
       谱聚类:   cluster.SpectralClustering
  降维任务：
       主成分分析: decomposition.PCA
       截断SVD和LSA： decomposition.TruncatedSVD
       字典学习:  decomposition.SparseCoder
       因子分析:  decomposition.FactorAnalysis
       独立成分分析: decomposition.FastICA
       非负矩阵分解: decomposition.NMF
       LDA:          decomposition.LatentDirichlet.Allocation
