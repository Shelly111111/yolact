import paddle.nn as nn
import paddle.nn.functional as F

class InterpolateModule(nn.Layer):
	"""
	This is a module version of F.interpolate (rip nn.Upsampling).
	Any arguments you give it just get passed along for the ride.
	"""

	def __init__(self, *args, **kwdargs):
		super().__init__()

		self.args = args
		self.kwdargs = kwdargs

	def forward(self, x):
		return F.interpolate(x, *self.args, **self.kwdargs)
