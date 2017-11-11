from math import exp
from math import pi
from  math import  sqrt

#定义高斯概率密度计算概率
def calculate_p(x, mean, variance):

    exponent = exp(-(x - mean)**2 / (2*variance))

    return ( exponent / (sqrt(2*pi) * sqrt(variance)) )


#计算每一类属性的每个值对于某一个类别的概率,并且合并概率
def calculate_attr_p(prior, predictset , mean , variance ):
    #计算属性值数目
    attr_num = len(predictset[0]) - 1
    lenth =  len(predictset)

    attr_p_list = []  #记录每一类的属性的概率
    for i in range(attr_num):
        l = []    #记录每一类属性的各个值对应的概率

        m = mean[i]      #每一类属性的均值和方差
        v = variance[i]
        for j in range(lenth):
            x = predictset[j][i]
            l.append(calculate_p(x,m,v))

        attr_p_list.append(l)

    #合并概率
    P = []
    for i in  range(lenth):     #i指示的是第i个样本
        temp = 1
        for j in range(len(attr_p_list)):    #j指示第j个属性
            temp *= attr_p_list[j][i]
        P.append(temp * prior)

    return P

#根据计算的概率对数据进行分类
def classify(predictset , m0, v0 , m1 , v1 , prior0, prior1):

    #分别计算每个样本为类别0，类别1的概率
    P0 = calculate_attr_p(prior0, predictset, m0, v0)
    P1 = calculate_attr_p(prior1, predictset, m1, v1)

    result = []   #记录每个样本被划分的类别
    for i in range(len(P1)):
        if P1[i] >= P0[i]:
            result.append(1)
        else:
            result.append(0)

    #计算准确率
    l = len(predictset)
    sum = 0
    for j in range(l):
        if predictset[j][-1] == result[j]:
            sum += 1

    accuracy = sum / l
    return accuracy


