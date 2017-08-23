#coding=utf-8

import numpy as np
import os
import operator

def img2vector(filename):
	vector = np.zeros((1,1024))
	with open(filename) as f:
		for i in range(32):
			line = f.readline()
			for j in range(32):
				vector[0,32*i+j] = int(line[j])
	return vector

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

def test_set(dir='./digits/testDigits'):
	# 测试集
	test_labels = []

	# 测试集文件目录
	test_files_list = os.listdir(dir)
	test_nums = len(test_files_list)
	test_sets = np.zeros((test_nums,1024))

	i = 0
	# 将训练集标签保存到hw_labels
	for file in test_files_list:
		test_labels.append(int(file[0]))
		test_sets[i,:] = img2vector(dir+file)
		i += 1
	return test_sets,test_labels

def test_file(file):
	# 测试集
	test_labels = []

	# 测试集文件目录
	test_sets = np.zeros((1,1024))

	# 将训练集标签保存到hw_labels
	test_labels.append(int(file[0]))
	test_sets[:] = img2vector(file)
	return test_sets,test_labels

def predict(inX,data_set,data_labels,k=10):
	data_set_size = data_set.shape[0]           # 获取训练集的行数,假设为m
	new_inX = np.tile(inX,(data_set_size,1))    # 创建一个行数为m的矩阵，每行数据与输入数据(测试数据)相同
	diff_matrix = new_inX - data_set            # 差矩阵(上面两矩阵相减)
	sq_diff_matrix = diff_matrix**2             # 平方
	distance = (sq_diff_matrix.sum(axis=1))**0.5    # 距离: 先求平方和，再开方
	sort_distance_index = distance.argsort()
	pre_labels = {}

	for i in range(k):
		label = data_labels[sort_distance_index[i]]
		pre_labels[label] = pre_labels.get(label,0) + 1
	sorted_pre_labels = sorted(pre_labels.iteritems(),key=lambda x:x[1],reverse=True)
	return sorted_pre_labels[0][0]

def score(pred_labels,test_labels):
	pred = np.array(pred_labels)
	test = np.array(test_labels)
	res = (pred==test).astype(int)
	return res.sum()*1.0/res.shape[0]

if __name__ == '__main__':
	import time
	start = time.time()
	print("获取训练集")
	traininig_sets , traininig_labels= traininig_set()
	print("获取测试集")
	test_sets , test_labels = test_set()
	pred_labels = []
	print("预测中...")
	for test in test_sets:
		pred_labels.append(predict(test,traininig_sets,traininig_labels,k=20))
	print ('总共用时%f秒\n'%(time.time()-start))
	print ("准确率为:")
	print(score(pred_labels,test_labels))
	with open('labels','w') as f:
		f.write('test:'+str(test_labels)+'\npred:'+str(pred_labels))