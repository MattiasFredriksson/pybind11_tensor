# Import test module
import sys
sys.path.append('../out/Release/pybind11_tensor_test/slice')
import slice

# Other Imports
import time
import unittest
import numpy as np
import psutil

class Test_slice_matrix():#unittest.TestCase):
	def setUpClass():
		Test_slice_matrix.rng = np.random.default_rng()

	#
	#   Helpers
	#

	def gen_random_tensor(self, shape=(10, 10, 10)):
		''' Generate random tensor of given shape.
		'''
		return Test_slice_matrix.rng.uniform(-1, 1, shape)

	def gen_randomized_tensor(self, rank=4, max_dims=100):
		''' Generate random rotation matrices on form Nx3x3
		'''
		shape = Test_slice_matrix.rng.integers(2, max_dims, rank)
		return Test_slice_matrix.rng.uniform(-1, 1, shape)


	#
	#   Tests
	#

	def test_A_slice_100(self):
		N = 100

		T = self.gen_random_tensor((N, N, N))
		T_out = slice.slice(T)

		assert np.allclose(T, T_out), 'slice_matrix<double, 3> failed'

	def test_C_slice_Nx3x5(self):
		N = 100

		T = self.gen_random_tensor((N, 3, 5))
		T_out = slice.slice(T)

		assert np.allclose(T, T_out), 'slice_matrix<double, 3> failed'

	def test_D_slice_Nx31x18(self):
		N = 10

		T = self.gen_random_tensor((N, 31, 18))
		T_out = slice.slice(T)

		assert np.allclose(T, T_out), 'slice_matrix<double, 3> failed'

	def test_E_slice_Nx3x5x4(self):
		N = 10

		T = self.gen_random_tensor((N, 3, 4, 5))
		T_out = slice.slice(T)

		assert np.allclose(T, T_out), 'slice_matrix<double, 4> failed'

	def test_F_slice_Nx3x5x4x7(self):
		N = 10000

		T = self.gen_random_tensor((N, 3, 4, 5, 7))

		t = time.time()
		T_out = slice.slice(T)
		#print('\n\n%f' % (time.time() - t))

		assert np.allclose(T, T_out), 'slice_matrix<double, 5> failed'

	def test_G_chip_Nx3x5x4x7(self):
		N = 10000

		T = self.gen_random_tensor((N, 3, 4, 5, 7))
		t = time.time()
		T_out = slice.chip(T)
		#print('\n\n%f' % (time.time() - t))

		assert np.allclose(T, T_out), 'chip-slice_matrix<double, 5> failed'

	def test_H_slice_Random(self):
		N = 1000

		for r in range(3, 5):
			for i in range(N):
				T = self.gen_randomized_tensor(rank=r, max_dims=33)
				T_out = slice.slice(T)
				assert np.allclose(T, T_out), 'slice_matrix<double, %i> failed on random tensor.' % r


class Test_slice_vector(unittest.TestCase):
	def setUpClass():
		Test_slice_vector.rng = np.random.default_rng()

	#
	#   Helpers
	#

	def gen_random_tensor(self, shape=(10, 10, 10)):
		''' Generate random tensor of given shape.
		'''
		return Test_slice_vector.rng.uniform(-1, 1, shape)

	def gen_randomized_tensor(self, rank=4, max_dims=100):
		''' Generate random rotation matrices on form Nx3x3
		'''
		shape = Test_slice_vector.rng.integers(2, max_dims, rank)
		return Test_slice_vector.rng.uniform(-1, 1, shape)


	#
	#   Tests
	#

	def test_A_slice_NxN(self):
		N = 100

		T = self.gen_random_tensor((N, N))
		T_out = slice.slice_vector(T)

		assert np.allclose(T, T_out), 'slice_vector<double, 2> failed'

	def test_B_slice_Nx5(self):
		N = 100

		T = self.gen_random_tensor((N, 5))
		T_out = slice.slice_vector(T)

		assert np.allclose(T, T_out), 'slice_matrix<double, 2> failed'

	def test_C_slice_NxNxN(self):
		N = 100

		T = self.gen_random_tensor((N, N, N))
		T_out = slice.slice_vector(T)

		assert np.allclose(T, T_out), 'slice_vector<double, 3> failed'

	def test_D_slice_Nx3x5(self):
		N = 100

		T = self.gen_random_tensor((N, 3, 5))
		T_out = slice.slice_vector(T)

		assert np.allclose(T, T_out), 'slice_matrix<double, 3> failed'

	def test_E_slice_NxNxNxN(self):
		N = 25

		T = self.gen_random_tensor((N, N, N, N))
		T_out = slice.slice_vector(T)

		assert np.allclose(T, T_out), 'slice_vector<double, 4> failed'

	def test_F_slice_Nx4x3x5(self):
		N = 100

		T = self.gen_random_tensor((N, 4, 3, 5))
		T_out = slice.slice_vector(T)

		assert np.allclose(T, T_out), 'slice_matrix<double, 4> failed'

	def test_G_slice_NxNxNxNxN(self):
		N = 10

		T = self.gen_random_tensor([N]*5)
		T_out = slice.slice_vector(T)

		assert np.allclose(T, T_out), 'slice_vector<double, 5> failed'

	def test_H_slice_Nx2x3x4x5(self):
		N = 50

		T = self.gen_random_tensor((N, 2, 3, 4, 5))
		T_out = slice.slice_vector(T)

		assert np.allclose(T, T_out), 'slice_matrix<double, 5> failed'
