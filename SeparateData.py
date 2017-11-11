from random import randint

#根据分割比例将数据集分为训练集和测试集
def separatedata(ratio , dataset = []):

    #ratio为训练集所占的比例
    all_rows = len(dataset)
    train_len =  int(all_rows * ratio)
    trainset = []
    while  len(trainset) < train_len:
        Index = randint(0, len(dataset)-1)
        trainset.append(dataset.pop(Index))
    #输出划分结果
    #print("Ratio: %f, Split %d rows into train = %d rows and predict = %d rows" % ( ratio,all_rows, len(trainset), len(dataset)) )

    return trainset, dataset      #dataset中剩下的就是测试集

