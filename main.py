#! -*- coding:utf-8 -*-

from keras.layers import Layer , Embedding , Flatten , Dense , Lambda , convolutional
import keras.backend as K 
from keras.models import Sequential
from keras.models import Model
from keras.layers import *
from keras.callbacks import Callback
from keras import regularizers
from keras.callbacks import Callback 
import keras
import numpy as np 
import os , sys
import math
sys.path.append("../")
from odelstm import *
import tensorflow as tf
import reader
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config=tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

np.random.seed(1234)
tf.set_random_seed(1234)



'''
dataset parameters
'''
embedding_dim = 300  # the dimension of word embedding
char_embedding_dim = 50  # the dimension of char embedding
s_maxlen = 53  # the maximun length of sentence
w_maxlen = 38  # the maximun length of word
num_class = 5
char_size = 42  

'''
model parameters
'''
emb_dropout =  0.2
char_emb_dropout = 0.2
input_dropout = 0.0
recurrent_dropout = 0.0
output_dropout = 0.5
scales = [5,10,15]   # the window size of each scale
scale_nums = [2,2,2]   # the number of small hidden states for each ODE-LSTM
units = 50    # the dimension of each small hidden state
l2 = 0.001
lamda = 0.01   # the balance factor of penalization loss
output_size = 150  # the size of MLP
filters = 50 # output dim of char embedding
width = 3  # kernel size of conv

''' 
train parameters
'''
epochs = 25
tune_epochs = 5
batch_size = 200
lr = 0.001
tune_lr = 1e-4


'''
Input
'''
data_root = 'data/sst5/embeddings/'
vector_path = data_root + 'glove.filtered.npz'
train_path = data_root +  'train_word2idx.json'
dev_path = data_root +  'dev_word2idx.json'
test_path = data_root + 'test_word2idx.json'
# load dataset
train_sentences , train_words , train_labels = reader.load_data_wc(train_path , shuffle = True)
dev_sentences , dev_words , dev_labels = reader.load_data_wc(dev_path , shuffle = True)
test_sentences , test_words , test_labels = reader.load_data_wc(test_path , shuffle = True)

# prepare the data format of the model
train_sen , train_word , train_label = reader.prepare_data_wc( train_sentences , train_words , train_labels , \
												 s_maxlen = s_maxlen , w_maxlen = w_maxlen)
dev_sen , dev_word , dev_label = reader.prepare_data_wc( dev_sentences , dev_words , dev_labels , \
												 s_maxlen = s_maxlen , w_maxlen = w_maxlen)
test_sen , test_word , test_label = reader.prepare_data_wc( test_sentences , test_words , test_labels , \
												 s_maxlen = s_maxlen , w_maxlen = w_maxlen)


# load word embedding
with open(vector_path , 'rb') as f:
	matrix = np.load(f)['embeddings']
vocab_size = matrix.shape[0]
print('vocab_size',vocab_size)


def reshape_backend(x , shape):
	return tf.reshape(x , shape = shape)


def reduce_max_backend(x , axis) :
	x = tf.reduce_max(x , axis = axis)
	return x

def concat_max_backend(x) :
	x = K.concatenate(x , axis = -1)
	x  = tf.reduce_max(x , axis = 1)
	return x

# penalization loss for ODE-LSTM
def regular_loss(layers , unit , scale_num , lamda, sparse=False) :

	def _loss(y_true , y_pred) :
		kernel_matrix = []
		regular_loss = 0
		for layer in layers :
			hidden_kernel = layer.recurrent_kernel[:,-(unit * scale_num):]
			# print(hidden_kernel.shape)
			hidden_kernel = tf.transpose(hidden_kernel)
			perscale_matrix = K.reshape(hidden_kernel , (scale_num , unit * unit)) 
			regular_loss += K.sum(K.square(K.dot(perscale_matrix , K.transpose(perscale_matrix)) - K.eye(scale_num)))
		if sparse:
			# task_loss = K.sparse_categorical_crossentropy(y_true,y_pred)
			task_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
		else:
			task_loss = K.categorical_crossentropy(y_true , y_pred)
		return task_loss + lamda * regular_loss
	return _loss

# Triple-S operation to speed up.
def reformat_input(data , scale ):
	batch_size , maxlen , d_in = data.shape
	print('shape of data : ' , batch_size , maxlen , d_in)
	idx = [i for j in range(scale-1,1000,1) for i in range(j-(scale-1),j+1,1)]
	x = tf.pad(data,((0,0),(scale-1,0),(0,0)), mode ='CONSTANT')  # [-1 , maxlen + scale -1 , d_in]
	print(' x shape : ' , x.shape)
	# print(idx)
	# print(np.array(idx[:scale * maxlen]))
	key = tf.gather(x , np.array(idx[:scale * maxlen] ) , axis = 1)
	print(' key shape : ' , key.shape)
	key = tf.reshape(key , (-1 , scale , d_in))  # [b x seq_len x ksize x d_model]
	return key


def model(scale , scale_num , unit , emb_dropout , char_emb_dropout, output_dropout):

	x_in = Input(shape=(s_maxlen,), dtype='int32' , name = 'word_input')
	char_in = Input(shape=(s_maxlen,w_maxlen,), dtype='int32' , name ='char_input')
	x = x_in
	x_char = char_in

	x = Embedding(input_dim = vocab_size , output_dim = embedding_dim , name  = 'embedding')(x)
	x_char = Embedding(input_dim = char_size , output_dim = char_embedding_dim , name = 'char_embedding')(x_char)

	x = Dropout(emb_dropout)(x)
	x_char = Dropout(char_emb_dropout)(x_char)

	# conv for char emb
	x_char = convolutional.Conv2D(filters = filters , kernel_size = (1,width) , activation = 'relu')(x_char)
	x_char = Lambda(reduce_max_backend , arguments = {'axis' : 2})(x_char)

	x = Concatenate(axis = -1)([x , x_char])

	branch_nums = len(scale)
	x_branch  = [ _ for _ in range(branch_nums)] 
	modelstm_branch = [ _ for _ in range(branch_nums)]
	output_branch = [ _ for _ in range(branch_nums)]

	ode_lstms = []

	for i in range(branch_nums) :
		
		# use triple-S operation to reshape the data from (batch_size,length,dim) to (batch_size,length,scale,dim) 
		keys = Lambda(reformat_input , arguments = {'scale' : scale[i]})(x)

		# start_time = time.time()

		modelstm_branch[i] = ODE_LSTM(units= unit , 
				scales = scale[i],
				scale_nums = scale_num[i] , 
				return_sequences= False , 
				input_dropout = input_dropout , 
				recurrent_dropout = recurrent_dropout)

		ode_lstms.append(modelstm_branch[i])

		output_branch[i] = modelstm_branch[i](keys)

		# reshape [batch * maxlen , scale_nums * units] --> [-1 , maxlen , scale_nums * units]
		output_branch[i] = Lambda(reshape_backend , arguments = {
									'shape':(-1 , s_maxlen , scale_num[i] * unit)})(output_branch[i])

		# end_time = time.time()
		# print("run time : %.2f s"%(end_time-start_time))


	if branch_nums != 1 :
		x = Lambda(concat_max_backend)(output_branch)
	else :
		x = Lambda(reduce_max_backend , arguments = {'axis' : 1})(output_branch[-1])

	x = Dropout(output_dropout)(x)

	# MLP
	x = Dense(output_size ,  activation = 'relu')(x)

	x = Dense(num_class, activation= 'softmax' ,  kernel_regularizer = regularizers.l2(l2))(x)

	model = Model([x_in,char_in] , x)
	model.summary()


	model.layers[4].set_weights([matrix])  # load pretrain word embedding
	model.layers[4].trainable = False    # freeze word embedding

	loss = regular_loss(ode_lstms , unit , scale_num[-1] , lamda)   # penalization loss
	model.compile(loss= loss , 
					optimizer= keras.optimizers.Adam(lr = lr) , 
					metrics = ['acc'])

	return loss , model

test_acc_list = []
max_test_acc = 0
save_path = None

class callback_test(Callback) :
	def __init__(self , x = None , y = None):
		self.x = x
		self.y = y

	def on_epoch_end(self , epoch , logs ={}):
		loss , acc = self.model.evaluate(self.x , self.y , verbose = 0) 
		print('loss : {} , test acc : {}'.format(loss,acc))
		test_acc_list.append(acc)
		# if acc > max_test_acc:
		# 	max_test_acc = acc
		# 	save_name = os.path.join(save_path , str(epoch) + '.h5')
		# 	self.model.save(save_name)
		# print(test_acc_list)


if __name__ == "__main__":

	print('parameters :')
	print('scale : ' + str(scales) + ' scale_num : ' + str(scale_nums) + ' unit : ' + str(units))
	print('emb_dropout : ' + str(emb_dropout) + ' char_emb_dropout :  ' + str(char_emb_dropout) + \
			' input_dropout : ' + str(input_dropout) + \
			' recurrent_dropout : ' + str(recurrent_dropout) + ' output_dropout : ' + str(output_dropout))
	print('\n')

	# save_path = str(char_emb_dropout)+'_'+str(emb_dropout)+'_'+str(units)

	# if not os.path.exists(save_path):
	# 	os.makedirs(save_path)

	loss , modelstm_model = model(scales, scale_nums , units , emb_dropout, char_emb_dropout , output_dropout )


	one_hot_label_train = keras.utils.to_categorical(train_label , num_class)
	one_hot_label_dev = keras.utils.to_categorical(dev_label , num_class)
	one_hot_label_test = keras.utils.to_categorical(test_label , num_class)

	history = modelstm_model.fit([train_sen, train_word], one_hot_label_train , 
								epochs = epochs ,
								batch_size = batch_size , 
								verbose = 2 , 
								validation_data = ([dev_sen , dev_word], one_hot_label_dev) ,
								callbacks = [callback_test([test_sen, test_word] , one_hot_label_test)]
								)

	# save_name = os.path.join(save_path , 'static.h5')
	# modelstm_model.save(save_name)
				
	acc_loss = history.history['loss']
	acc = history.history['acc']
	val_loss = history.history['val_loss']
	val_acc = history.history['val_acc']

	'''
	fine-tune stage
	'''

	modelstm_model.layers[4].trainable = True

				
	modelstm_model.compile(loss= loss, 
					optimizer= keras.optimizers.Adam(lr = tune_lr) , 
					metrics = ['acc'])


	history = modelstm_model.fit([train_sen, train_word], one_hot_label_train , 
								epochs = tune_epochs ,
								batch_size = batch_size , 
								verbose = 2 , 
								validation_data = ([dev_sen , dev_word], one_hot_label_dev) ,
								callbacks = [callback_test([test_sen, test_word] , one_hot_label_test)]
							)

	# save_name = os.path.join(save_path , 'finetune.h5')
	# modelstm_model.save(save_name)

	acc_loss += history.history['loss']
	acc += history.history['acc']
	val_loss += history.history['val_loss']
	val_acc += history.history['val_acc']

	train_max_acc = max(acc)
	dev_max_acc = max(val_acc)
	dev_max_index = val_acc.index(dev_max_acc)
	test_acc_dev = test_acc_list[dev_max_index]


	print('train_max_acc : %f , dev_max_acc : %f ; test_acc_dev :%f' \
			% ( train_max_acc , dev_max_acc , test_acc_dev))


	K.clear_session()
	tf.reset_default_graph()





