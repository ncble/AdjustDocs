
import keras
import tensorflow
### For Delta Orthogonal initialization (tensorflow)
from keras.initializers import Initializer
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

class ConvolutionDeltaOrthogonal(Initializer):
	"""Initializer that generates a delta orthogonal kernel for ConvNets.
	The shape of the tensor must have length 3, 4 or 5. The number of input
	filters must not exceed the number of output filters. The center pixels of the
	tensor form an orthogonal matrix. Other pixels are set to be zero. See
	algorithm 2 in [Xiao et al., 2018]: https://arxiv.org/abs/1806.05393
	Args:
	gain: Multiplicative factor to apply to the orthogonal matrix. Default is 1.
	  The 2-norm of an input is multiplied by a factor of 'sqrt(gain)' after
	  applying this convolution.
	seed: A Python integer. Used to create random seeds. See
	  @{tf.set_random_seed} for behavior.
	dtype: The data type.
	"""
	def __init__(self, gain=1.0, seed=None, dtype=dtypes.float32):
		self.gain = gain
		self.dtype = self._assert_float_dtype(dtypes.as_dtype(dtype))
		self.seed = seed

	def __call__(self, shape, dtype=None, partition_info=None):
		if dtype is None:
			dtype = self.dtype
		# Check the shape
		if len(shape) < 3 or len(shape) > 5:
			raise ValueError("The tensor to initialize must be at least three-dimensional and at most five-dimensional")

		if shape[-2] > shape[-1]:
			raise ValueError("In_filters cannot be greater than out_filters.")

		# Generate a random matrix
		a = random_ops.random_normal([shape[-1], shape[-1]], dtype=dtype, seed=self.seed)
		# Compute the qr factorization
		q, r = gen_linalg_ops.qr(a, full_matrices=False)
		# Make Q uniform
		d = array_ops.diag_part(r)
		q *= math_ops.sign(d)
		q = q[:shape[-2], :]
		q *= math_ops.sqrt(math_ops.cast(self.gain, dtype=dtype))
		if len(shape) == 3:
			weight = array_ops.scatter_nd([[(shape[0]-1)//2]], array_ops.expand_dims(q, 0), shape)
		elif len(shape) == 4:
			weight = array_ops.scatter_nd([[(shape[0]-1)//2, (shape[1]-1)//2]],
										array_ops.expand_dims(q, 0), shape)
		else:
			weight = array_ops.scatter_nd([[(shape[0]-1)//2, (shape[1]-1)//2,
										(shape[2]-1)//2]],
										array_ops.expand_dims(q, 0), shape)
		return weight

	def get_config(self):
		return {"gain": self.gain, "seed": self.seed, "dtype": self.dtype.name}

	def _assert_float_dtype(self, dtype):
		"""Validate and return floating point type based on `dtype`.
		`dtype` must be a floating point type.
		Args:
		  dtype: The data type to validate.
		Returns:
		  Validated type.
		Raises:
		  ValueError: if `dtype` is not a floating point type.
		"""
		if not dtype.is_floating:
			raise ValueError("Expected floating point type, got %s." % dtype)
		return dtype