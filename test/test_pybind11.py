# Import test module
import sys
sys.path.append('../out/Release/pybind11_tensor_test')

# Other Imports
import pybind11_tensor_test as Test
import numpy as np
import os


A = np.arange(72).reshape((6, 3, 4)).astype(np.float32)
A_c = A.copy()
print(type(A))
print(A.flags.writeable)
print(A.flags.c_contiguous)


print('\n-------\nadd_self_map-stride[::2, ::2, ::2]:')

Test.add_self_map(A[::2, ::2, ::2])

assert np.allclose(A[::2, ::2, ::2] - A_c[::2, ::2, ::2], A_c[::2, ::2, ::2]), 'add_self-stride[::2, ::2, ::2] failed'
assert not np.allclose(A - A_c, A_c), 'add_self-stride[::2, ::2, ::2] failed: All changed'
A = A_c.copy()
print("Success")

print('\n-------\nadd_self_map-stride[::2, ::2]:')

Test.add_self_map(A[::2, ::2])
assert np.allclose(A[::2, ::2] - A_c[::2, ::2], A_c[::2, ::2]), 'add_self-stride[::2, ::2] failed'
assert not np.allclose(A - A_c, A_c), 'add_self-stride[::2, ::2] failed: All changed'
A = A_c.copy()
print("Success")

print('\n-------\nadd_self:')

B = Test.add_self(A)
assert np.allclose(B-A, A), 'add_self failed'
assert np.allclose(A, A_c), 'add_self failed: changed A'
print("Success")

print('\n-------\nadd_self-stride:')

B = Test.add_self(A[::2])

assert np.allclose(B-A[::2], A[::2]), 'add_self-stride failed'
assert np.allclose(A, A_c), 'add_self-stride failed: changed A'
print("Success")

print('\n-------\nprint-const:')

Test.print_const(A)
print("Success")

print('\n-------\nprint:')

Test.print(A[2::2, :, :])
print("Success")

print('\n-------\nprint-const: (non-writeable)\n')

A.flags.writeable = False
Test.print_const(A)
print("Success")

print('\n-------\nprint-fortran: (should fail for now)\n')

Test.print(np.asfortranarray(A))
