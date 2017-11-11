

def  loaddata(path):

    #先将文件中的数据都存储到一个列表中
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            newline = line.split(',')
            l = []
            for i in range(len(newline)):
                if i != len(newline ) - 1:
                    l.append(float(newline[i]))
                else:
                    l.append(float(newline[i].split('\n')[0]))
            dataset.append(l)
    return dataset
