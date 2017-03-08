---
title: 机器学习--手写数字识别(python实现)
tags: python,机器学习,KNN
grammar_cjkRuby: true
---
### **一、图片转字符**
要实现手写数字识别，那么首先需要将手写的数字转变成可读写的字符，这样才能使用KNN算法来进行后续的步骤。所以首先看看如何实现图片转字符：

首先需要用到pillow库，怎么安装的这里就不再详述了。
代码参考：[用Python把图片变成字符画][1]。

需要注意的地方：

- 1.为了便于后续计算，所以最后只用0和1来显示图片，即 **ascii_char = list("10")**
- 2.**使用方法**：
命令行模式下：
```p
python pic2char.py **.png -o **.txt --width=32 --height=32
```
> 之所以高宽要设为**32**，是因为我在学《机器学习实战》课程中给出的训练集是32*32，所以代码都是以32为基准写的；

这种命令行操作模式下，只能每次操作一张图片很麻烦，所以你可以将代码改一下即可。怎么改的不详述，给个思路吧：
1. 使用os.listdir获取图片文件
2. for循环依次将图片文件转为字符文件即可

**pic2char.py**
```python
#-*- coding:utf-8 -*-

from PIL import Image
import argparse

#命令行输入参数处理
parser = argparse.ArgumentParser()

parser.add_argument('file')     #输入文件
parser.add_argument('-o', '--output')   #输出文件
parser.add_argument('--width', type = int, default = 32) #输出字符画宽
parser.add_argument('--height', type = int, default = 32) #输出字符画高

#获取参数
args = parser.parse_args()

IMG = args.file
WIDTH = args.width
HEIGHT = args.height
OUTPUT = args.output

ascii_char = list("10")

# 将256灰度映射到70个字符上
def get_char(r,g,b,alpha = 256):
    if alpha == 0:
        return ' '
    length = len(ascii_char)
    gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)

    unit = (256.0 + 1)/length
    return ascii_char[int(gray/unit)]

if __name__ == '__main__':

    im = Image.open(IMG)
    im = im.resize((WIDTH,HEIGHT), Image.NEAREST)

    txt = ""

    for i in range(HEIGHT):
        for j in range(WIDTH):
            txt += get_char(*im.getpixel((j,i)))
        txt += '\n'

    print txt

    #字符画输出到文件
    if OUTPUT:
        with open(OUTPUT,'w') as f:
            f.write(txt)
    else:
        with open("output.txt",'w') as f:
            f.write(txt)
```
代码使用示例：
**图片：**

![3](http://p1.bpimg.com/567571/5fe9e93b99462637.png)

**转化字符：**

![图片转字符](http://p1.bqimg.com/567571/494d949db27af036.png)

### **二、KNN算法实现**
现在假设我们已经生成了足够多的训练集和测试集。

![训练集](http://p1.bqimg.com/567571/19e3108b37f0231c.png)

#### **1.将每个文件数据转化成一维向量**
每个txt文件包含的事32 * 32的数据，我们可以转化成 1 * 1024的向量，实现代码：
```python
def img2vector(filename):
    vector = np.zeros((1,1024))
    with open(filename) as f:
        for i in range(32):
            line = f.readline()
            for j in range(32):
                vector[0,32*i+j] = int(line[j])
    return vector
```


#### **2.获取训练集/测试集数据和标签**
我的训练集数据路径是 **'./digits/testDigits'**,你需要将路径改成你的对应路径。
```python
def traininig_set(dir='./digits/testDigits'):
    # 训练集
    training_labels = []
    
    # 训练集文件目录
    trainig_files_list = os.listdir(dir)
    trainig_nums = len(trainig_files_list)
    trainig_sets = np.zeros((trainig_nums,1024))
    
    i = 0
    # 将训练集标签保存到hw_labels
    for file in trainig_files_list:
    	training_labels.append(int(file[0]))
    	trainig_sets[i,:] = img2vector(dir+file)
    	i += 1
    return  trainig_sets,training_labels
```
测试集数据的获取同理。

#### **3.预测**
步骤：

- 根据距离计算公式计算测试数据与每一个训练数据的距离(**欧几里得距离**)
- 将距离**由小到大**排序，并获取排序后对应的序号集
- 统计各距离对应的训练集的标签，并排序
- 根据统计的标签，选出**标签数最多**的即为预测值
```python
def predict(inX,data_set,data_labels,k=10):
    # 获取训练集的行数,假设为m
    data_set_size = data_set.shape[0]      
    
    # 创建一个行数为m的矩阵，每行数据与输入数据(测试数据)相同     
    new_inX = np.tile(inX,(data_set_size,1))    
    
    # 差矩阵(上面两矩阵相减)
    diff_matrix = new_inX - data_set         
    
    # 平方
    sq_diff_matrix = diff_matrix**2   
    
    # 距离: 先求平方和，再开方
    distance = (sq_diff_matrix.sum(axis=1))**0.5  
    
    # 将距离由小到大排序，并返回对应的序号集
    sort_distance_index = distance.argsort()
    
    pre_labels = {}
    for i in range(k):
    	label = data_labels[sort_distance_index[i]]
    	pre_labels[label] = pre_labels.get(label,0) + 1
    	
    sorted_pre_labels = sorted(pre_labels.iteritems(),key=lambda x:x[1],reverse=True)
    return sorted_pre_labels[0][0]
```

#### **4.查看准确度**
传入预测标签和测试标签，分别转为np.array对象后可以很方便的进行比较并得出结果。
```python
def score(pred_labels,test_labels):
    pred = np.array(pred_labels)
    test = np.array(test_labels)
    res = (pred==test).astype(int)
    return res.sum()*1.0/res.shape[0]
```

```python
print("获取训练集")
traininig_sets , traininig_labels= traininig_set()
print("获取测试集")
test_sets , test_labels = test_set()
pred_labels = []
print("预测中...")
for test in test_sets:
	pred_labels.append(predict(test,traininig_sets,traininig_labels,k=20))
print ("准确率为:")
print(score(pred_labels,test_labels))
```

<br><br><hr>
<footer style="padding:10px;border-radius:10px;;text-align:center;background-color:rgb(11,12,13);color:white;">
written by <b style="color:tomato;font-size:16px;">MARSGGBO</b>
<br><span style="font-size:16px;">
2017-2-14</span>
</footer>


  [1]: http://www.jianshu.com/p/991cb07b3ad3