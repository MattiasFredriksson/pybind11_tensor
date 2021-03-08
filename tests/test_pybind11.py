# Import test module
import sys
sys.path.append('../out/Release/pybind11_tensor_test/transfer')
import transfer

# Other Imports
import numpy as np
import os
import psutil


A = 1 + np.arange(72).reshape((6, 3, 4)).astype(np.float64)
A_c = A.copy()
M = np.arange(10000).reshape((100, 100)).astype(np.float64)
T = np.arange(10000).reshape((100, 10, 10)).astype(np.float64)

def alloc_dummy_A(N=10):
    for i in range(N):
        tmp = A.copy()

print(type(A))
print(A.dtype)
print("Writeable:\t", A.flags.writeable)
print("C contiguous:\t", A.flags.c_contiguous)

transfer.check_eigen_move_operator()

print('\n-------\nadd_self:')

B = transfer.add_self(A)
assert np.allclose(B-A, A), 'add_self failed'
assert np.allclose(A, A_c), 'add_self failed: changed A'
print("Success")

print('\n-------\nadd_self-stride:')

B = transfer.add_self(A[::2])

assert np.allclose(B-A[::2], A[::2]), 'add_self-stride failed'
assert np.allclose(A, A_c), 'add_self-stride failed: changed A'
print("Success")


print('\n-------\nadd_self_map:')

transfer.add_self_map(A)

assert np.allclose(A - A_c, A_c), 'add_self_map failed'
A = A_c.copy()
print("Success")


print('\n-------\nadd_self_map-stride[::2, ::2]:')

transfer.add_self_map(A[::2, ::2])
assert np.allclose(A[::2, ::2] - A_c[::2, ::2], A_c[::2, ::2]), 'add_self_map-stride[::2, ::2] failed'
assert not np.allclose(A - A_c, A_c), 'add_self_map-stride[::2, ::2] failed: All changed'
A = A_c.copy()
print("Success")


print('\n-------\nadd_self_map-stride[::2, ::2, ::2]:')

transfer.add_self_map(A[::2, ::2, ::2])

assert np.allclose(A[::2, ::2, ::2] - A_c[::2, ::2, ::2], A_c[::2, ::2, ::2]), 'add_self_map-stride[::2, ::2, ::2] failed'
assert not np.allclose(A - A_c, A_c), 'add_self_map-stride[::2, ::2, ::2] failed: All changed'
A = A_c.copy()
print("Success")


print('\n-------\nadd_self_map_ret:')

B = transfer.add_self_map_ret(A)

#assert B is A, 'add_self_map_ret failed: Same object' #todo: how to determine if different python objectss
assert not B.flags.owndata, 'add_self_map_ret failed: New object'
assert np.allclose(B - A, 0), 'add_self_map_ret failed: Different objects'
assert np.allclose(B - A_c, A_c), 'add_self_map_ret failed'
A = A_c.copy()
print("Success")

print('\n-------\nadd_self_ref:')

B = transfer.add_self_ref(A)
assert np.allclose(B-A, A), 'add_self_ref failed'
assert np.allclose(A, A_c), 'add_self_ref failed: changed A'
print("Success")


print('\n-------\nadd_self_ref-stride:')

B = transfer.add_self_ref(A[::2])

assert np.allclose(B-A[::2], A[::2]), 'add_self_ref-stride failed'
assert np.allclose(A, A_c), 'add_self_ref-stride failed: changed A'
print("Success")

try:
    print('\n-------\nadd_self_ref_ref:')

    B = transfer.add_self_ref_ref(A)
    assert np.allclose(B - A, A), 'add_self_ref_ref failed'
    assert np.allclose(A, A_c), 'add_self_ref_ref failed: changed A'
except TypeError as e:
    print("Success, TensorRef<TensorRef<...>> is not a convertible type.")


try:
    print('\n-------\nadd_self_ref_undef:')

    iters = 0
    while iters < 1000:
        B = transfer.add_self_ref_undef(A)
        alloc_dummy_A()
        assert np.allclose(B - A, A), 'add_self_ref_undef failed'
        assert np.allclose(A, A_c), 'add_self_ref_undef failed: changed A'
        iters += 1
    print("Failed reaching undefined behavior.") # Only consistent if DEBUG_ALLOC is defined.

except Exception as e:
    print(e)
    print("Successfully reached undefined behavior at iteration", iters)

print('\n-------\nadd_self_repeat10:')

B = transfer.add_self_repeat10(A)
assert np.allclose(B-A[np.newaxis, :, :], A[np.newaxis, :, :]), 'add_self_repeat10 failed'
assert np.allclose(A, A_c), 'add_self_repeat10 failed: changed A'
print("Success")

print('\n-------\nadd_self_vector:')

V = [A for i in range(100)]
B = transfer.add_self_vector(V)
assert np.allclose(np.subtract(B, A[np.newaxis, :, :, :]), A), 'add_self_vector failed'
assert np.allclose(V, A_c[np.newaxis, :, :, :]), 'add_self_vector failed: changed V'
print("Success")

print('\n-------\nadd_self_vector-memcleanup:')

mem = psutil.virtual_memory().used
N = [M for i in range(100)]
for i in range(1000):
    N = transfer.add_self_vector(N)
    W = transfer.add_self_vector(V)
mem_diff = psutil.virtual_memory().used - mem
assert mem_diff < 10 * mem, 'add_self_vector-memcleanup failed, used %i more bytes' % mem_diff
print("Success")


print('\n-------\nadd_self_repeat10-memcleanup:')

mem = psutil.virtual_memory().used
for i in range(1000):
    O = transfer.add_self_repeat10(M)
    W = transfer.add_self_repeat10(T)
mem_diff = psutil.virtual_memory().used - mem
assert mem_diff < 10 * mem, 'add_self_repeat10-memcleanup failed, used %i more bytes' % mem_diff
print("Success")


print('\n-------\nprint-const:\n')

transfer.print_const(A)
print("Success")

print('\n-------\nprint:\n')

transfer.print(A[2::2, :, :])
print("Success")

print('\n-------\nprint-const: (non-writeable)\n')

A.flags.writeable = False
transfer.print_const(A)
print("Success")


if __name__ == "__main__":

    print('\n-------\nprint-fortran: (should fail for now)\n')

    transfer.print(np.asfortranarray(A))
