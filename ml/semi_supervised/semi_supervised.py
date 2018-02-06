# https://arxiv.org/pdf/1602.06797.pdf

import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import random
from munkres import Munkres

vocab_path = '/home/anson/ml/semi supervised clustering for short text via deep representation/ml/cnn/runs/1508064243/vocab'
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

def getInputFeed(text_arr):
	# Map data into vocabulary
	arr = np.array(list(vocab_processor.transform(text_arr)))
	return arr


#load the model
checkpoint_dir = "/home/anson/ml/semi supervised clustering for short text via deep representation/ml/cnn/runs/1508064243/checkpoints"
allow_soft_placement = True
log_device_placement = False


checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        #output dims
        dimensions = graph.get_operation_by_name("output_features").outputs[0]

labelled_text = []
labels = []
#load the supervised data
for root, dirs, files in os.walk('./../../data/supervised/'):
	for file in files:
		print(file)
		labels.append(file.split('.')[0])
		with open('./../../data/supervised/'+file,'r') as f:
			labelled_text.append(f.readlines())

print('shape of labelled text')
print(len(labelled_text))

#load unsupervised text
unlabelled_text = []
for root, dirs, files in os.walk('./../../data/unsupervised/'):
	for file in files:
		with open('./../../data/unsupervised/'+file,'r') as f:
			unlabelled_text += f.readlines()

print('shape of unlabelled text')
print(len(unlabelled_text))

#compute f for each text
f_text = []
text_arr = []
for text in labelled_text:
	text_arr += text

text_arr += unlabelled_text
print('text arr')
print(len(text_arr))
print(text_arr)

f_text = sess.run(dimensions, {input_x: getInputFeed(text_arr), dropout_keep_prob: 1.0})

print(f_text.shape)
num_classes = 4

f_min = 10000000
f_max = -10000000

for i, dim in enumerate(f_text):
	min_val = min(dim)
	max_val = max(dim)
	if min_val<f_min:
		f_min = min_val
	if max_val > f_max:
		f_max = max_val

################################################################################################
#randomly initialize the centroids
################################################################################################
Uk = []
for i in range(num_classes):
	tmp = []
	for j in range(384):
		tmp.append(random.random()*f_max + random.random()*f_min)
	Uk.append(tmp)

################################################################################################
#cluster assignment
################################################################################################
m = Munkres()
def assignClusterForSupervised(f,labels,Uk):
	mat = []
	#compute the matrix of the average distance of the label to the centroids
	avg_dist = np.zeros((num_classes,len(Uk)))
	num_samples = [0 for _ in range(len(Uk))]
	for j,label in enumerate(labels):
		avg_dist_tmp_arr = []
		num_samples[label] += 1
		for i, centroid in enumerate(Uk):
			avg_dist[label][i] += np.linalg.norm(np.array(centroid)-np.array(f[j]))

	for i in range(num_classes):
		avg_dist[i] /= num_samples[i]

	#find the closest centroids to each label using hungarian algorithm
	min_cost_indexes = m.compute(avg_dist)

	min_cost = [0 for _ in range(len(Uk))]
	for item in min_cost_indexes:
		min_cost[item[0]] = item[1]

	rk = []
	for j,label in enumerate(labels):
		rk.append(min_cost[label])

	return rk


def assignClusterForUnsupervised(f,Uk):
	rk = []
	# print("Uk")
	# print(len(Uk))
	for dim in f:
		min_dist = 100000000
		rk_tmp = 0
		for centroid_num, centroid in enumerate(Uk):
			dist = np.linalg.norm(np.array(centroid)-np.array(dim))
			if min_dist > dist:
				min_dist = dist
				rk_tmp = centroid_num
		rk.append(rk_tmp)

	return rk

#get rk
#the supervised text is of the form [[text_label1_1,text_label1_2,....],[text_label2_1,text_label2_2,....],.........]
#the unsupervised text is of the form [text1, text2, .......]
#Uk is of the form [[centroid1_dim1,centroid1_dim2,....],[centroid2_dim1,centroid2_dim2,....],......]
#sess is the tensorflow session
def assignCluster(f,labels,Uk):
	f_sup = f[:len(labels)]
	f_unsup = f[len(labels):]

	rk_sup = assignClusterForSupervised(f_sup,labels,Uk)
	
	dim_arr_unsup = sess.run(dimensions, {input_x: getInputFeed(text_arr), dropout_keep_prob: 1.0})
	rk_unsup = assignClusterForUnsupervised(dim_arr_unsup,Uk)

	return rk_sup+rk_unsup
################################################################################################
#estimate the centroid
################################################################################################
#s: supervised text
#n: nth text
#j: jth centroid
#ug: centroid array for labeled texts
#f: dense representation of text
#u: centroid array for all texts
#l: margin
#k: kth centroid
#K: number of centroids/classes
#gn: centroid num for nth labeled text
#G: array of centroid num for labeled text
#N: total number of text samples
#L: number of labeled samples
#r: r[n][k] is 1 if kth cluster is assigned to nth sample and 0 otherwise

alpha = 0.01
l = 0.1

def delta1(x):
	if x!=0:
		return 1
	else:
		return 0
def delta2(x1,x2):
	if x1==x2:
		return 1
	else:
		return 0
def delta_dash(n,j,ugn,fn,u,l):
	return delta1(l+(np.linalg.norm(fn-ugn))**2-(np.linalg.norm(fn-u[j]))**2)

def I1(k,gn):
	return delta2(k,gn)
def I2(k,j,n,ugn,fn,u,l):
	return delta2(k,j)*delta_dash(n,j,ugn,fn,u,l)
def I3(k,j,n,ug,fn,u,l):
	return (1-delta2(k,j))*delta_dash(n,j,ug,fn,u,l)

def w(n,k,ug,f,u,l,G):
	gn = G[n]

	sum1 = 0
	for j in range(num_classes):
		if j==gn:
			continue
		sum1 += I2(k,j,n,ug,f[n],u,l)

	sum2 = 0
	for j in range(num_classes):
		if j==gn:
			continue
		sum2 += I3(k,j,n,ug,f[n],u,l)

	return (1-alpha)*(I1(k,gn)+sum1-sum2)


def estimate_centroid(k,f,u,l,G,N,L,r):
	num1 = 0
	for n in range(N):
		num1 += alpha*r[n][k]*f[n]	

	num2 = 0
	for n in range(L):
		ug = G[n]
		num2 += w(n,k,ug,f,u,l,G)*f[n]	

	den1 = 0
	for n in range(N):
		den1 += alpha*r[n][k]

	den2 = 0
	for n in range(L):
		ug = G[n]
		den2 += w(n,k,ug,f,u,l,G)

	return (num1+num2)/(den1+den2)

################################################################################################
#update parameter
################################################################################################
# def getCost(r,u,G,f,N,K,L):
# 	term1 = 0
# 	for i in range(N):
# 		for k in range(K):
# 			term1 = r[n][k]*(np.linalg.norm(f[n]-u[k]))**2
# 	term1 *= alpha

# 	term2 = 0
# 	for n in range(L):
# 		gn = G[n]
# 		ugn = u[gn]
# 		term2_0 += (np.linalg.norm(f[n]-u[k]))**2

# 		sum_2 = 0
# 		for j in range(K):
# 			if j==gn:
# 				continue
# 			sum2 += max(l+(np.linalg.norm(f[n]-ugn))**2-(np.linalg.norm(f[n]-u[j]))**2, 0)

# 		term2_0 += sum2

# 	term2 = tf.scalar_mul((1-alpha),term2_0)

# 	cost = tf.add(term1,term2)
# 	return term1+term2

#r is an nxk tensor
#f1 is a kxn tensor of ||f-uk||^2
#f2 is a 1xL tensor of ||f-ugn||^2
#m is an LxK tensor of max(l+||f-ugn||^2-||f-uk||^2,0)
def getCost():
	r = tf.placeholder(tf.float32, [None, num_classes])
	uk = tf.placeholder(tf.float32, [num_classes, 384])
	ugn = tf.placeholder(tf.float32, [None,384])
	num_labelled_data = tf.placeholder(tf.int32, shape=())

	f1 = tf.square(tf.norm(tf.concat([[tf.subtract(dimensions,uk[i])] for i in range(num_classes)],axis=0),axis=-1))
	dimensions_labelled = tf.slice(dimensions, [0,0], [num_labelled_data,-1])
	f2 = tf.expand_dims(tf.square(tf.norm(tf.subtract(dimensions_labelled,ugn), axis=-1)), axis=0)

	m = tf.placeholder(tf.float32, [None, num_classes])

	t1 = tf.scalar_mul(alpha,tf.reduce_sum(tf.matmul(r,f1),[0,1]))
	t2 = tf.scalar_mul((1-alpha),tf.reduce_sum(f2,[0,1]))
	t3 = tf.reduce_sum(m,[0,1])
	t4 = tf.add(tf.add(t1,t2),t3)

	return t4, r, uk, ugn, m, num_labelled_data

def getRmat(r,num_classes):
	result = np.zeros((len(r),num_classes))
	# print(r)
	for i,cluster in enumerate(r):
		result[i,cluster] = 1

	return result

def getf1mat(f,Uk):
	n = len(f)
	k = len(Uk)
	result = np.zeros(k,n)
	for i in range(n):
		for j in range(k):
			result[i,j] = np.linalg.norm(np.array(f[i])-np.array(Uk[j]))**2

	return result

def getf2mat(f,Uk,L,labels,r):
	f_labelled = f[:L]
	result = np.zeros((1,L))
	for i in range(L):
		ugn = Uk[r[i]]
		result[i] = np.linalg.norm(np.array(f[i])-np.array(ugn))**2

	return result

def getmmat(l,f,Uk,L,labels,r):
	k = len(Uk)
	result = np.zeros((L,k))
	for i in range(L):
		gn = r[i]
		ugn = Uk[gn]
		for j in range(k):
			if j==gn:
				continue

			result[i,j] = max(l+(np.linalg.norm(f[i]-ugn))**2-(np.linalg.norm(f[i]-Uk[j]))**2, 0)

	return result

#########################################################################################
#train
#########################################################################################
checkpoint_save_every = 100
num_epochs = 400

def getBatch(labelled_text,unlabelled_text,sess,batch_size=128):
	num_supervised_samples = int(batch_size/8)
	per_class_samples_sup = int(num_supervised_samples/len(labelled_text))
	num_unsupervised_samples = batch_size-num_supervised_samples 

	if per_class_samples_sup==0:
		raise ValueError('The number of samples per class is zero. Reduce the number of classes or increase the batch size.')


	count_unlabelled = 0
	taken_all_samples = False
	while not taken_all_samples:
		labelled = []
		unlabelled = []
		labels = []
		for i in range(len(labelled_text)):
			for j in range(min(per_class_samples_sup, len(labelled_text[i]))):
				labelled.append(labelled_text[i][random.randint(0,len(labelled_text[i])-1)])
				labels.append(i)

		for i in range(len(unlabelled_text)):
			start = count_unlabelled
			end = min(count_unlabelled + num_unsupervised_samples, len(unlabelled_text))
			unlabelled = unlabelled_text[start:end]

		count_unlabelled = min(count_unlabelled + num_unsupervised_samples, len(unlabelled_text))

		yield(labelled, labels, unlabelled)


		if count_unlabelled == len(unlabelled_text):
			taken_all_samples = True

batch_size = 128
with graph.as_default():
	with sess.as_default():
		with tf.variable_scope('unsupervised_vars'):
			cost,r1_mat,uk_placeholder,ugn_placeholder,m_mat,num_labelled_data  = getCost()
			# train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			optimizer = tf.train.AdamOptimizer(name="adam_for_clustering").minimize(cost)
			all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
			vars_to_train = [v for v in all_vars if v.name.find('unsupervised_vars')>-1]
			#initialize above variables
			for v in vars_to_train:
				sess.run(v.initializer)

			# print(all_vars)
			for epoch in range(num_epochs):
				batch = getBatch(labelled_text, unlabelled_text, sess, batch_size=batch_size)
				for labelled, labels, unlabelled in batch:
					text = []
					text += labelled
					text += unlabelled

					f = sess.run(dimensions, {input_x: getInputFeed(text), dropout_keep_prob: 1.0})
					#assign cluster
					r = assignCluster(f,labels,Uk)
					r_mat = getRmat(r, num_classes)
					# print("r")
					# print(r)
					# print("f")
					# print(f)
					G = r[:len(labels)]
					N = len(text)
					L = len(labels)
					K = num_classes
					#estimate centroid
					for k in range(len(Uk)):
						Uk[k] = estimate_centroid(k,f,Uk,l,G,N,L,r_mat)

					Ugn = []
					for index in range(len(G)):
						Ugn.append(Uk[r[index]])
					#update parameters
					_, cost_val = sess.run([optimizer, cost], {
							r1_mat: getRmat(r,num_classes),
							uk_placeholder: Uk,
							ugn_placeholder: Ugn, 
							# f1_mat: getf1mat(f,Uk),
							# f2_mat: getf2mat(f,Uk,len(labels),labels,r),
							m_mat: getmmat(l,f,Uk,L,labels,r),
							input_x: getInputFeed(text), 
							dropout_keep_prob: 1.0,
							num_labelled_data: L
						})
					print('Cost: ' + str(cost_val))

				if epoch == checkpoint_save_every:
					path = saver.save(sess, './../model', global_step=epoch)
					print('Saving checkpoint in ' + path)

path = saver.save(sess, './../model', global_step=epoch)
print('Saving checkpoint in ' + path)

# write results to a file
r_count = 0
cluster_map = {}
cluster_map_reverse = {}
classified_clusters = {}
# label_arr = []
# centroid_label_map = []
label_num = 0
for root, dirs, files in os.walk('./../../data/supervised/'):
	for file in files:
		cur_label = file.split('.')[0]
		with open('./../../data/supervised/'+file,'r') as f:
			labelled_text_test = f.readlines()
			# label_arr.append(cur_label)
			# centroid_label_map.append(r[r_count])
			cluster_map[r[r_count]] = {'label': cur_label, 'label_num': label_num}
			cluster_map_reverse[cur_label] = label_num
			classified_clusters[cur_label] = []

			r_count += len(labelled_text_test)
			label_num += 1

unlabelled_text_test = []
for root, dirs, files in os.walk('./../../data/unsupervised/'):
	for file in files:
		with open('./../../data/unsupervised/'+file,'r') as f:
			print(file)
			unlabelled_text_test += f.readlines()

f = sess.run(dimensions, {input_x: getInputFeed(unlabelled_text_test), dropout_keep_prob: 1.0})
#assign cluster
r = assignClusterForUnsupervised(f,Uk)
print(r)
r_count = 0
for text in unlabelled_text_test:
	classified_clusters[cluster_map[r[r_count]]['label']].append(text)

	r_count += 1

for label_vals, text_arr in classified_clusters.items():
	with open('./../../test/'+label_vals+'.txt', 'w') as f:
		f.write('\n'.join(text_arr))

