import pandas as pd
TR = pd.read_csv(r"train.txt",sep = "\t",header=0)
#读取同一文件夹路径下的txt文件，故不加路径
#r是转义符，防止文件路径的\被转义；不加r时可以把路径的\改为/。本段代码不体现r的作用kkk
#sep=','以逗号为分隔符。如txt中1，2，3，python读出为1 2 3
#header=None txt的第一行数据作为python读出的第一行数据
#header=0 txt第一行数据作为python读出数据的表格索引；header=1 txt的第二行数据作为索引

groups=TR.groupby(TR.y)
group_1=groups.get_group(1)
group_0=groups.get_group(0)
#画出散点图
ax=group_0.plot.scatter(x='X1',y='X2',color='r')
group_1.plot.scatter(x='X1',y='X2',color='b',ax=ax)