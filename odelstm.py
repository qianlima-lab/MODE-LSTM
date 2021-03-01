#! -*- coding:utf-8 -*-

from keras.layers import Layer
import keras.backend as K 
import tensorflow as tf  


# multi-kernel of one scale RNN
class ODE_LSTM(Layer):
	'''
	units : the hidden size of each kernel
	scales : the window size(length) of each kernel
	scale_nums : the number of kernels
	'''
	def __init__(self,
				units,
				scales,
				scale_nums,
				return_sequences=False,
				input_dropout =None,
				recurrent_dropout = None , 
				kernel_regularizer = None ,
				**kwargs):
		self.units = units
		self.scales = scales
		self.scale_nums = scale_nums
		self.return_sequences = return_sequences
		self.input_dropout = input_dropout
		self.recurrent_dropout = recurrent_dropout
		self.kernel_regularizer = kernel_regularizer
		super(ODE_LSTM,self).__init__(**kwargs)

	# initialization parameters
	def build(self, input_shape):
		input_dim = input_shape[-1]
		# recurrent_dim = self.units * self.scale_nums 
		self.kernel = self.add_weight(
			shape =(input_dim, self.units * self.scale_nums  * 4 ) , name = 'kernel',
			initializer = 'glorot_uniform')
		self.recurrent_kernel = self.add_weight(
			shape = (self.units , self.units * self.scale_nums  * 4) , name = 'recurrent_dim',
			initializer = 'orthogonal')
		self.bias = self.add_weight(
			shape = (self.units * self.scale_nums  * 4,) , name = 'bias',
			initializer = 'zeros')
		self.built = True
		if self.input_dropout :
			self._kernel = K.dropout(self.kernel , self.input_dropout)
			self._kernel = K.in_train_phase(self._kernel , self.kernel)
		else :
			self._kernel = self.kernel
		if self.recurrent_dropout :
			self._recurrent_kernel = K.dropout(self.recurrent_kernel , self.recurrent_dropout)
			self._recurrent_kernel = K.in_train_phase(self._recurrent_kernel , self.recurrent_kernel)
		else :
			self._recurrent_kernel = self.recurrent_kernel

	def one_step(self, inputs , states):
		# c_last & h_last shape : [batch , units * scale_nums]
		x_in , (c_last , h_last) = inputs , states
		# input --> hidden , [batch , scale_nums * units * 4]
		x_ih = K.dot(x_in , self._kernel)

		'''
		reshape h & recurrent_weight for parallel computing
		reshape h : [scale_nums , batch , 1 , units]
		reshape recurrent_weight : [scale_nums , 1 , 4 * units , units]
		'''
		h_per_kernel  = tf.split(h_last , num_or_size_splits = self.scale_nums , axis = 1)
		h_per_kernel = tf.expand_dims(h_per_kernel , 2)

		recurrent_weight = K.reshape(self._recurrent_kernel , 
									shape = (self.scale_nums , 1 , 4 * self.units , self.units))
			
		# hidden --> hidden , [batch , 4 * scale_nums * units]
		# x_hh : [scale_nums , batch , 4*units ]
		x_hh = tf.reduce_sum(h_per_kernel * recurrent_weight , 3 )
		# each element of list x_hh is shape [1 , batch , 4*units]
		x_hh = tf.split(x_hh , num_or_size_splits = self.scale_nums , axis = 0)
		# x_hh : [1 , batch , 4 * scale_nums * units]
		x_hh = tf.concat(x_hh , axis = 2)
		# x_hh : [batch , 4*scale_nums*units]
		x_hh = tf.squeeze(x_hh , axis = 0)
		x_out = K.bias_add(x_ih + x_hh , self.bias)

		# version 1
		dim_num = self.scale_nums * self.units
		f_gate = K.sigmoid(x_out[: , :dim_num])
		i_gate = K.sigmoid(x_out[: , dim_num : dim_num * 2])
		o_gate = K.sigmoid(x_out[: , dim_num * 2 : dim_num * 3])
		cell_state = K.tanh(x_out[:,-dim_num:])

		c_out = f_gate * c_last + i_gate * cell_state
		h_out = o_gate * K.tanh(c_out)
		

		# h_out & c_out : scale_nums * units
		return h_out , [c_out , h_out]


	def call(self, inputs):
		initial_states = [
			K.zeros((K.shape(inputs)[0] , self.units * self.scale_nums)),
			K.zeros((K.shape(inputs)[0] , self.units * self.scale_nums))
		]
		outputs = K.rnn(self.one_step , inputs , initial_states)
		if self.return_sequences :
			return outputs[1]
		else :
			return outputs[0]

	def compute_output_shape(self , input_shape):
		if self.return_sequences :
			return (input_shape[0] , input_shape[1] , self.units * self.scale_nums)
		else :
			return (input_shape[0] , self.units * self.scale_nums)