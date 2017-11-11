from copy import deepcopy
from LoadData import loaddata
from SeparateData import separatedata
from Splitbyclass import splitbyclass
from Predict import classify
import numpy as np
import matplotlib.pyplot as plt
import LearnfromTrainset


#在main.py 中是计算特定划分比例下的算法准确率，在本次中则讨论0.5-0.9之间的划分比例与准确率的关系



if __name__ == '__main__':


    #给出数据路径，并将数据存入列表
    Path = 'pima-indians-diabetes.data'
    Dataset = loaddata(Path)

    Ratio_list = np.arange(0.5, 0.9, 0.01)
    Accuracy_list = []
    for j in range(0,1000):

        for i in range(len(Ratio_list)):

            tmp = deepcopy(Dataset)     #最开始使用了浅拷贝即：tmp = Dataset,导致数据集越划分越小，出现方差 均值为0之类的情况
            # 划分数据集
            Ratio = Ratio_list[i]
            Trainset, Predictset = separatedata(Ratio, tmp)

            #把训练集按照类别划分
            Class0 , Class1 = splitbyclass(Trainset)


            #计算先验概率
            Prior0 , Prior1 = LearnfromTrainset.prior_p(Class0, Class1)

            #计算每一个类别中每种属性的均值与方差
            M0 , V0 = LearnfromTrainset.meanAndvariance(Class0)
            M1 , V1 = LearnfromTrainset.meanAndvariance(Class1)

            #利用学习到的知识进行预测
            accuracy = classify(Predictset, M0, V0, M1, V1, Prior0, Prior1)
            if j == 0:
                #第一组测试的结果先插入列表
                Accuracy_list.append(accuracy)
            else:
                Accuracy_list[i] += accuracy    #把同一个比例下的测试准确率相加，最后求均值

    for m in range(len(Accuracy_list)):
        Accuracy_list[m] /= 1000     #经过1000次测试之后，计算每一个比例对应的测试准确率

    plt.title('The Accuracy - Ratio of trainset')
    plt.xlabel('Ratio')
    plt.ylabel('Accuracy')
    max_i = np.argmax(Accuracy_list)  # 最大值的下标


    plt.plot(Ratio_list, Accuracy_list,'m-+',linewidth = 2.0)

    #标出图中的最大值
    plt.plot(Ratio_list[max_i], Accuracy_list[max_i],'kD')
    show_max = '[' + str(Ratio_list[max_i]) + ' ' + str(Accuracy_list[max_i]) + ']'
    plt.annotate(show_max, xytext=(Ratio_list[max_i], Accuracy_list[max_i]), xy=(Ratio_list[max_i], Accuracy_list[max_i]))
    plt.savefig('result.png', dpi=600)
    plt.show()


