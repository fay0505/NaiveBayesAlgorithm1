

#计算先验概率，以及每个属性值的均值，以及方差

def prior_p(class_0 = [], class_1 = []):
    len0 = len(class_0)
    len1 = len(class_1)
    p0 = len0 / (len1 + len0)
    p1 = len1 / (len1 + len0)

    return p0, p1

def meanAndvariance(data = []):   #计算某一个类别中所有属性的均值和方差
    attr_num = len(data[0]) - 1   #计算属性的个数
    len_data = len(data)

    Mean = []
    Variance = []

    for i in range(attr_num):
        sum = 0
        for j in range(len_data):
            sum += data[j][i]
        m = sum / len_data              #计算均值
        Mean.append(m)

    for i in range(attr_num):
        sum = 0
        for j in range(len_data):
            sum += (data[j][i] - Mean[i]) ** 2
        v = (sum / (len_data - 1))     #计算方差
        #print(v)
        Variance.append(v)

    return Mean, Variance
