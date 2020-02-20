import numpy as np

class DataInput:
	def __init__(self, data, batch_size):
		self.batch_size = batch_size
		self.data = data
		self.epoch_size = len(self.data) // self.batch_size
		if self.epoch_size * self.batch_size < len(self.data):
			self.epoch_size += 1
		self.i = 0

	def get_batch(self):

		ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
		self.i += 1
		if self.i >= self.epoch_size:
			self.i = 0

		return ts

