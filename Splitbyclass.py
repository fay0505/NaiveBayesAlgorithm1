#将训练集中的数据按照类别的属性分成两部分

def splitbyclass(trainset = []):
    class_0 = []
    class_1 = []

    for i in range(len(trainset)):
        if trainset[i][-1] == 0:
            class_0.append(trainset[i])
        else:
            class_1.append(trainset[i])


    return class_0, class_1