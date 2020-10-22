#降维任务（无监督学习）
   PCA降维（主成分分析法）
   #PCA
      定义：最常用的一种降维方法，通常用于高位数据集的可视化，还可用作
            数据的压缩与预处理

      理解：PCA可以吧具有相关性的高维变量合称为线性无关的低维变量，称为
            主成分，主成分能够尽可能保留原始数据的信息

      相关术语：方差：反应随机变量取值与期望的平均偏离程度，离差平方和
                协方差：衡量两个随机变量之间的线性相关程度，方差的推广。反应平均变化趋势
                协方差矩阵：变量的协方差值构成的矩阵，协方差为0则线性无关。对称矩阵
                特征向量：反应矩阵变换的特征向量。Aa=ca,a为特征向量
                
      原理：矩阵的主成分就是其协方差矩阵对应的特征向量，按照特征值大小进行排序
            最大的特征值为第一主成分，其次第二主成分，以此类推。周志华（机器学习）
            样本集D={x1,x2,x3,x4,x5...xm}#每个样本x1都可以是一个多维的向量，本质是把向量的一个行字段看做是随机变量
                  低维空间维数d
            过程： 1.所有样本进行中心化xi-1/m(sum(x1-xm))#消除样本间的差异性（量纲和数值量级不同），中心变为原点。还可以除以标准差使其去除量纲.使矩阵的特征向量变为可以描述的数据方向
                   2.计算样本的协方差矩阵 XX'
                   3.对协方差矩阵XX’进行特征值分解
                   4.取出最大的d个特征值所对应的的特征向量w1....wd
                输出：投影矩阵W=（w1,w2,w3...wd）
                
            #本质其实是在高位平面选择一个方向，将所有数据都投影到低维空间，又让地位空间
            #的信息尽可能保证。直观的想法是，让投影后的信息尽可能地分散，保留最大的信息
            #使用数学方法找到这个投影的矩阵
            二：
               由于标准化了均值，所以是原点为中心。var(a)=1/m sum(ai^2)
               于是上面的问题被形式化表述为：寻找一个一维基，使得所有数据变换为这个基上的坐标表示后，方差值最大。
            三。
                对于二维降成一维，只需要找到使方差最大的方向的斜线就可以了，但对于多维降维
                我们需要考虑更多。二维降到一维只需要考虑一个方向，三维降到二维需要两个方向
                如果我们还是单纯只选择方差最大的方向，很明显，这个方向与第一个方向应该是“几乎重合在一起”，显然这样的维度是没有用的
                因此，应该有其他约束条件。从直观上说，让两个字段尽可能表示更多的原始信息，我们是不希望它们之间存在（线性）相关性的
                因为相关性意味着两个字段不是完全独立，必然存在重复表示的信息。
                
           四。由于均值为0，cov(a,b)=1/m sum(ai*bi)
               在两个向量均值为0的情况下，相互的协方差就是两者的内积除以元素数m。
               为了让协方差为0，我们选择第二个基时只能在与第一个基正交的方向上选择。因此最终选择的两个方向一定是正交的。至此，我们得到了降维问题的优化目标：将一组N维向量降为K维
               （K大于0，小于N），其目标是选择K个单位（模为1）正交基，使得原始数据变换到这组基上后，各字段两两间协方差为0，而字段的方差则尽可能大（在正交的约束下，取最大的K个方差）。

           五。设我们有m个n维数据记录，将其按列排成n乘m的矩阵X，设C=1/mXXT，则C是一个对称矩阵，其对角线分别个各个字段的方差，而第i行j列和j行i列元素相同，表示i和j两个字段的协方差。

           
           六。协方差矩阵的对角化
               设原始数据矩阵X对应的协方差矩阵为C，而P是一组基按行组成的矩阵，设Y=PX，则Y为X对P做基变换后的数据。设Y的协方差矩阵为D，我们推导一下D与C的关系：
               D=====1/mYYT=1/m(PX)(PX)T=1/mPXXT=PTP(1/mXXT)PT=PCPT
               我们需要找的就是能够使原始协方差矩阵对角化的矩阵P
               优化目标变成了寻找一个矩阵P，满足PCPT是一个对角矩阵，并且对角元素按从大到小依次排列
               那么P的前K行就是要寻找的基，用P的前K行组成的矩阵乘以X就使得X从N维降到了K维并满足上述优化条件。
               #原始矩阵X经过矩阵P的变换之后变成矩阵Y，Y的协方差矩阵D为我们需要的对角化的协方差为0的矩阵，所以P就是我们所需要的变换矩阵。要第降到第几维就用P的前多少行

          七。求解能够让X的协方差矩阵对角化的矩阵P
              由于协方差阵C是实对称矩阵，满足特征向量为正交矩阵的特点。
              一个n行n列的实对称矩阵一定可以找到n个单位                                                                                                                                            正交特征向量。
              由特征向量构成矩阵E。由特征值分解可得 C=EAE^T，即E^TCE=A，A为C的特征值矩阵
              所以，P=E^T,即原矩阵X的协方差矩阵C的特征向量矩阵的转置。
              P是协方差矩阵的特征向量单位化后按行排列出的矩阵，其中每一行都是C的一个特征向量。
              如果设P按照Λ中特征值的从大到小，将特征向量从上到下排列，则用P的前K行组成的矩阵乘以原始数据矩阵X，就得到了我们需要的降维后的数据矩阵Y。
              
      #sklearn中的PCA
            在sklearn中，可以使用sklearn.decomposition.PCA
            加载PCA降维实例
            主要参数： n_components 指定主成分个数，即降维后的数据维度
                       svd_solver  设置特征值分解的方法，默认为auto



      #使用实例
             对鸢尾花数据降维和进行可视化
             鸢尾花数据集是个四维（萼片与花瓣的长宽）的数据，一共有三类样本。使用PCA实现对鸢尾花数据的降维
             实现在二维平面的可视化

             实例程序编写：
             import matplotlib.pyplot as plt
             from sklearn.decomposition import PCA
             from sklearn.datasets import load_iris
             2.加载数据进行降维
                data=load_iris()#字典形式加载鸢尾花数据集
                y=data.target#y表示数据集中的标签
                x=data.data#x表示数据集中的属性数据

                pca=PCA(n_components=2)#加载PCA算法，设置降维后主成分数目为2
                reduce_x=pca.fit_transform(x)#对原始数据进行降维，保存在reduce_x中
                        #所有运算返回的都是ndarray数组
            3.按类别保存降维后的数据
                #鸢尾花有三类样本，红，绿，蓝，分别设置三个序列保存
                red_x,red_y=[],[]
                blue_x,blue_y=[],[]
                green_x,green_y=[],[]

                #按鸢尾花的类别将降维之后的点保存在列表中,y[i]代表了0，1,2三个颜色的鸢尾花类别
                for i in range(len(reduce_x)):
                    if y[i]==0:
                        red_x.append(reduce_x[i][0])
                        red_y.append(reduce_x[i][1])
                    elif y[i]==1:
                        blue_x.append(reduce_x[i][0])
                        blue_y.append(reduce_x[i][1])
                    elif y[i]==2:
                        green_x.append(reduce_x[i][0])
                        green_y.append(reduce_x[i][1])

            4.降维可视化数据点
                  #使用pyplot中的散点图，对三类鸢尾花数据点进行展示
                  #一定要使用c='r'这样的形式来指明颜色
                  plt.scatter(red_x,red_y,c='r',marker='x')
                  plt.scatter(blue_x,blue_y,c='b',marker='D')
                  plt.scatter(green_x,green_y,c='g',marker='.')
                  plt.show()
                  
            总结：进行降维之后，数据依旧能够清晰的分为三类，这样不但可以
                  消减数据的维度，降低分类任务的工作量，还能保证分类的质量




       #NMF方法降维（非负矩阵分解）
            NMF（非负矩阵分解）：
            定义：NMF是在矩阵中所有元素均为非负数的约束条件下的矩阵分解方法
            性质：给定一个非负矩阵V，NMF找到非负矩阵W,非负矩阵H，使得
                  W和H的乘积近似等于矩阵V中的值，W矩阵的行数等于原矩阵行数
                  NMF能够广泛的应用于图像分析，文本挖掘，语音处理等领域
            分解矩阵：
                 W矩阵：基础图像矩阵，相当于从原矩阵V中抽取出来的特征
                 H矩阵：系数矩阵

            W矩阵为原始图像抽取的特征，H是其系数。

            原理：矩阵分解优化目标：最小化W矩阵与H矩阵的乘积和原矩阵V的差别
                   argmin1/2|X-WH|^2=1/2SUM(X-WH)^2  #基于欧氏距离的求解
                   argminJ(W,H)=sum(xlnx/wh-x+wh)#基于KL散度的求解（相对熵）
                   信息熵，是随机变量或整个系统的不确定性。熵越大，随机变量或系统的不确定性就越大。
                   相对熵，用来衡量两个取值为正的函数或概率分布之间的差异。
                   交叉熵，用来衡量在给定的真实分布下，使用非真实分布所指定的策略消除系统的不确定性所需要付出的努力的大小。
                   相对熵=交叉熵-信息熵
                   #NMF算法详解 http://blog.csdn.net/acdreamers/article/details/44663421

      #sklearn中NMF实例
            sklearn中，可以使用sklearn.decomposition.NMF加载NMF算法
                   主要参数：
                            n_components :指定分解之后矩阵单个维度
                            init:W,H矩阵初始化方式，默认为'nndsvdar'
                            #其他参数参考官网API
            2.实战目标，使用NMF对人类特征进行提取
                   sklearn中提供了olivetti人类数据，一共400个，大小为64*64
                   由于NMF分解得到的W矩阵等于是从原始矩阵中提取的特征，那么就
                   可以使用NMF对400个人类的数据进行特征提取
                   #通过设置k的大小，我们可以设置提取特征的数目，本实验中设置
                   #为k=6，随后将提取的特征通过图像展示出来

            3.实例程序编写

                1.导入相关包
                import matplotlib.pyplot as plt
                from sklearn import decomposition
                from sklearn.datasets import fetch_olivetti_faces#加载人脸数据
                from numpy.random import RandomState#用于创建随机数种子

                2.基本参数设置
                n_row,ncol=2,3 #即我们最后展示2行3列的特征提取。设置图像展示时的排列情况
                n_components = n_row*n_col #设置提取特征的数目
                image_shape=(64*64) #设置人脸数据图片的大小

                dataset = fetch_olivetti_faces(shuffle=True,random_state=RandomState(0))#dataset.images为400*64*64的图片数据。daaset.data为400*4096的二维矩阵，对应了图片降维之后的数据
                faces = datasets.data#加载数据，打乱数据。
                
                3.用函数的方法定义图像画图展示方式
                def plot_gallery(title,images,n_col=n_col,n_row=n_row):
                    plt.figure(figsize=(2.*n_col,2.26*n_row))#创建图片，并指定图片的大小（英寸）
                    plt.suptitle(title,size=16)   #设置标题与字号的大小


                    for i,comp in enumerate(images):
                        plt.subplot(n_row,n_col,i+1)
                        vmax = max(comp.max(),-comp.min())#选出comp中绝对值最大的

                        plt.imshow(comp.reshape(image_shape),cmap=plt.cm.gray,
                                   interpolation='nearest',vmin=-vmax,vmax=vmax)#对数值归一化，以灰度图的方式显示

                        plt.xticks(())
                        plt.yticks(())
                    plt.subplots_adjust(0.01,0.05,0.99,0.93,0.04,0.)#调整子图位置和间隔

                4.创建特征提取对象NMF，使用PCA进行对比
                    estimators=[('PCA',decomposition.PCA(n_components=6,whiten=True)),('NMF',decomposition.NMF(n_components=6,init='nndsvda'))]
                    #创建PCA于NMF的实例，放在一个列表中

                5.降维后数据点的可视化
                    for name,estimator in estimators:
                        estimator.fit(faces) #调用PCA或NMF提取特征
                        components_=estimator.components_#获得提取特征

                        plot.gallery(name,components_[:n_components])#调用函数，规定格式进行派讯

                        plt.show() #进行可视化
                        

#无监督学习聚类分割图像
        图像分割：利用图像灰度，颜色，文理，形状等特征，把图像分割成若干个互不重叠的区域
                  使其在同一区域呈现相似性，不同区域呈现差异性。然后就可以将分割图像中
                  独特性质区域取出来做不同的研究。

        常用方法：阈值分割：对图像的灰度值进行度量，设置不同类别的阀值，分成不同类别，达到分割目的
                  边缘分割：对图像的边缘进行检测，即检测图像灰度值发生跳变的地方，则为一片区域边缘
                  直方图法：对图像的颜色建立直方图，直方图波峰波谷表示一快区域颜色值范围，达到分割目的

                  特定理论：基于聚类分析，小波变换等理论完成图像分割

       #实例：利用Kmeans算法对图像进行像素点颜色进行聚类，完成简单的图像分割
        输出：同一聚类中的点使用相同颜色标记，不同聚类颜色不同

        技术路线 sklearn.cluster.Kmeans

        #实例代码
        1.建立工程并导入相关的包
        2.加载图片进行预处理
        3.加载Kmeans聚类算法
        4.对所有像素点进行聚类并输出
        #实例中涉及到了图像的加载和创建，因此我们使用PIL模块。

        1.建立工程并导入sklearn
          创建Kmeans.py
          导入sklern相关包
          import numpy as np
          import PIL.Image as image

        2.加载图片并进行预处理
          加载训练数据
          def loadData(filePath):
              f=open(filePath,'rb')#二进制形式打开文件
              data=[]

              img=image.open(f)                 #列表形式返回图片像素值
              m,n=img.size                     #获得图片的大小
              for i in range(m):              #将每个像素的RGB范围处理到0-1范围并存进data
                  for j in range(n):
                      x,y,z = img.getpixel((i,j))
                      data.append([x/256.,y/256.0,z/256.0])
              f.close()
              return np.mat(data),m,n         #以矩阵的形式返回data以及图片的大小


          imgData,row,col =loadData('kemeas/a.jpg')

          3.加载Kmeans算法
            km=KMeans(n_cluster=3)                 #指定聚类中心的个数
s


          4.依据聚类中心，对属于同一聚类的点使用同样的颜色进行标记
            lalbel = km.fit_predict(imageData)
            label = label.reshape([row,col])

            #创建一张新的灰度图保存聚类后的结果
            pic_new = image.new('L',(row,col))
            #根据类别向图片中添加灰度值
            for i in rang(row):
                for j in range(col):
                    pic_new.putpixel((i,j),256/(label[i][j]+1))
            #JPEG格式保存图像
            pic_new.save("result-4.jpg","JPEG")
            
        #注意：设置不同的K值，得到不同的聚类结果。K值的不确定性也是KMEANS方法的缺点。
        #往往需要多次尝试不同的K值。而层次聚类的方法就无需指定K值。只要给限制条件，就能直接得到类别数K。
                        
                   
