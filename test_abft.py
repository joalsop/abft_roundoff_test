import pytest
#import tcast
import torch
import random
import statistics

seed=123
torch.manual_seed(seed)
random.seed(seed)

# execution parameters
fp32_sum = True # perform accumulation and output C in fp32. Still downcast after presum, before gemm
use_rand = True # turn off to use a constant value for input elements (helpful for debugging)
rand_scale = 1.0 # rand generates normal distribution with stdev=1. scale for larger/smaller range
use_mod = True  # whether or not to use modulo arithmetic
modval = 1.0    # if using modulo arithmetic, which mod value to use

# absolute and relative abft error tolerance thresholds for counting above-threshold violations
rtol = 0.01
atol = 0.01

epsilon = 0.0001 # avoid divide by zero issues

def mod_gemm(A, B, C):
    M = C.shape[0]
    N = C.shape[1]
    K = A.shape[1]

    block = min(M, N, 4)
    dtype_accum = torch.float32 if fp32_sum else A.dtype

    Agemm = A.to(dtype_accum)
    Bgemm = B.to(dtype_accum)
    for blockm in range(0, M, block):
        for blockn in range(0, N, block):
            accum = torch.zeros(block, block, dtype=dtype_accum)
            for m in range(0, block):
                for n in range(0, block):
                    for k in range(K):
                        #accum = (accum + (A[blockm+m,k]*B[k,blockn+n]) % modval) % modval
                        product = Agemm[blockm+m, k]*Bgemm[k, blockn+n]
                        #product = product % (modval if product >=0 else -1*modval)
                        product = product % modval
                        accum[m,n] = accum[m,n] + product
                        #accum[m,n] = accum[m,n] % (modval if accum[m,n] >= 0 else -1*modval)
                        accum[m,n] = accum[m,n] % modval
                        #print("    +%s(%s*%s) mod %s = %s" % (A[blockm+m, k]*B[k, blockn+n], A[blockm+m, k], B[k, blockn+n],modval, accum[m,n]))
                    #print("done with %d, %d (accum is %s)" % (m, n, accum[m,n]))
            C[blockm:blockm+block,blockn:blockn+block] = accum.to(C.dtype)

def mod_sum2d(Xin, dim=0, use_mod=True):
    dtype_accum = torch.float32 if fp32_sum else X.dtype
    X = Xin.to(dtype_accum)
    dim0 = X.shape[0]
    dim1 = X.shape[1]
    if dim==0:
        retvec = torch.zeros(dim1, dtype=dtype_accum)
        for i in range(dim1):
            accum = 0
            for j in range(dim0):
                #print("    %s mod %s = %s" % (accum + X[j,i].item(), modval, (accum + X[j,i].item()) % modval))
                accum = (accum + X[j,i])
                if use_mod:
                    accum = accum % modval
            retvec[i] = accum
    else:
        retvec = torch.zeros(dim0, dtype=dtype_accum)
        assert(dim==1)
        for i in range(dim0):
            accum = 0.0
            for j in range(dim1):
                accum = (accum + X[i,j])
                if use_mod:
                    accum = accum % modval
            retvec[i] = accum
    return retvec

def get_mod_diffs(X, Y, modval):
    diff1 = abs(X-Y)
    diff2 = abs(X + -1*Y+modval)
    diff3 = abs(Y + -1*X+modval)
    return torch.minimum(torch.minimum(diff1, diff2), diff3)

def test_matrix_mul(m, n, k, datatype_str):
    abft_num_diff = 0
    abft_num_rdiff = 0
    abft_max_diff = 0.
    abft_max_rdiff = 0.

    datatype = getattr(torch, datatype_str)
    accum_dtype = torch.float32 if fp32_sum else datatype
    if use_rand:
        if "float" in datatype_str:
            A = torch.randn(m, k).float()*rand_scale
            B = torch.randn(n, k).float().T*rand_scale
        else:
            assert("int" in datatype_str)
            A = torch.randint(low=0, high=100, size=(m, k)).float()*int(rand_scale)
            B = torch.randint(low=0, high=100, size=(n, k)).float().T*int(rand_scale)
    else:
        # this can help with debugging
        A = torch.ones(m, k).float()/100
        B = torch.ones(n, k).float().T/100
    if use_mod:
        # I don't think this is necessary for correctness, but simplifies debugging
        # since it restricts the modulo arithmetic to positive numbers
        base_val = 0.001
        A = abs(A) % modval
        B = abs(B) % modval
    C = torch.zeros(m, n).float()
    postcm = torch.zeros(n-1).float()
    postcn = torch.zeros(m-1).float()

    A = A.to(datatype) #.float()
    B = B.to(datatype) #.float()
    C = C.to(accum_dtype) #.float()
    postcm = postcm.to(datatype) #.float()
    postcn = postcn.to(datatype) #.float()

    ## tensor cast - originally started using this library, but since it does not
    ## model roundoff error, we stick with builtin python types/math
    #datatype_tc = tcast.datatype(datatype_str)
    #A = tcast.cast(A, dtype=datatype_tc, roundmode="even")
    #B = tcast.cast(B, dtype=datatype_tc, roundmode="even")
    #C = tcast.cast(C, dtype=datatype_tc, roundmode="even")
    #if not fp32_sum:
    #    postcm = tcast.cast(postcm, dtype=datatype_tc, roundmode="even")
    #    postcn = tcast.cast(postcn, dtype=datatype_tc, roundmode="even")

    A = A
    B = B
    C = C

    #print("A is\n%s\nB is\n%s\nC is\n%s\n" % (A, B, C))

    # pre-checksum
    if use_mod:
        # modulo arithmetic is not associative over floating point multplication
        # (e.g., 0.5*10 mod 3 != 0.5*(10 mod 3)
        # therefore, only apply modulo after A*B multiplication
        A[-1,:] = mod_sum2d(A[0:-1,:], dim=0, use_mod=False)
        B[:,-1] = mod_sum2d(B[:,0:-1], dim=1, use_mod=False)
    else:
        if fp32_sum:
            A[-1,:] = torch.sum(A[0:-1,:].to(torch.float32), dim=0).to(A.dtype)
            B[:,-1] = torch.sum(B[:,0:-1].to(torch.float32), dim=1).to(B.dtype)
        else:
            A[-1,:] = torch.sum(A[0:-1,:], dim=0)
            B[:,-1] = torch.sum(B[:,0:-1], dim=1)

    #print("After presum A is\n%s\nB is\n%s\n" % (A, B))

    # GEMM
    try:
        if use_mod:
            mod_gemm(A, B, C)
        else:
            if fp32_sum:
                C = (A.to(torch.float32) @ B.to(torch.float32)).to(C.dtype)
            else:
                C = A @ B
    except OverflowError:
        # note: overflow may not throw this error, but should result in NaN
        print("overflow occurred in gemm!")
    #print("After GEMM, C is %s\n" % (C))

    # find diff between last row and sum of other rows
    last_row = C[-1,0:-1]
    try:
        if use_mod:
            postcm = mod_sum2d(C[0:-1,0:-1], dim=0, use_mod=True)
        else:
            postcm = torch.sum(C[0:-1,0:-1], dim=0)
    except OverflowError:
        print("overflow occurred in postcm!")
        exit(1)
    if use_mod:
        diffs=get_mod_diffs(postcm, last_row, modval)
    else:
        diffs=abs(postcm - last_row)
    rdiffs=diffs/torch.maximum(abs(last_row), epsilon*torch.ones(last_row.shape[0]))
    max_diff = diffs.max().item()
    max_rdiff = rdiffs.max().item()
    diff_count = sum(1 for d in diffs if d > atol)
    rdiff_count = sum(1 for d in rdiffs if d > rtol)
    abft_max_diff = max(max_diff, abft_max_diff)
    abft_max_rdiff = max(max_rdiff, abft_max_rdiff)
    abft_num_diff += diff_count
    abft_num_rdiff += rdiff_count
    #if diff_count > 0.5*len(diffs):
    #    print("WARNING: more than half of elements exceed max abs diff. diffs are:\n%s" % diffs)
    #    print("last row:\n%s" % last_row)
    #    print("postcm:\n%s" % postcm)
    #if rdiff_count > 0.5*len(rdiffs):
    #    print("WARNING: more than half of elements exceed max rel diff. rdiffs are:\n%s" % rdiffs)
    #    print("last row:\n%s" % last_row)
    #    print("postcm:\n%s" % postcm)

    # find diff between last col and sum of other cols
    last_col = C[0:-1,-1].to(datatype)
    try:
        if use_mod:
            postcn = mod_sum2d(C[0:-1,0:-1], dim=1, use_mod=True)
        else:
            postcn = torch.sum(C[0:-1,0:-1], dim=1)
    except OverflowError:
        # note: overflow may not throw this error, but should result in NaN
        print("overflow occurred in postcn!")
        exit(1)
    if use_mod:
        diffs=get_mod_diffs(postcn, last_col, modval)
    else:
        diffs=abs(postcn - last_col)
    rdiffs=diffs/torch.maximum(abs(last_col), epsilon*torch.ones(last_col.shape[0]))
    max_diff = diffs.max().item()
    max_rdiff = rdiffs.max().item()
    diff_count = sum(1 for d in diffs if d > atol)
    rdiff_count = sum(1 for d in rdiffs if d > rtol)
    abft_max_diff = max(max_diff, abft_max_diff)
    abft_max_rdiff = max(max_rdiff, abft_max_rdiff)
    abft_num_diff += diff_count
    abft_num_rdiff += rdiff_count
    #if diff_count > 0.5*len(diffs):
    #    print("WARNING: more than half of elements exceed max abs diff. diffs are:\n%s" % diffs)
    #    print("last col:\n%s" % last_col)
    #    print("postcn:\n%s" % postcn)
    #if rdiff_count > 0.5*len(rdiffs):
    #    print("WARNING: more than half of elements exceed max rel diff. rdiffs are:\n%s" % rdiffs)
    #    print("last col:\n%s" % last_col)
    #    print("postcn:\n%s" % postcn)

    return (abft_max_diff, abft_max_rdiff, abft_num_diff, abft_num_rdiff)

gemm_sizes = [
        #2,
        #4,
        8,
        16,
        32,
        64,
        # setting use_mod for below sizes it will take a while
        #128,
        #256,
        #512,
        #1024,
        #2048,
        #4096,
        #8192,
        #16384,
        ]

datatypes = [
        #'int32', # mainly for testing
        'float32',
        'float16',
        'bfloat16',
        # currently 8b types don't work with pytorch, at least for mi210
        #'float8_e5m2',
        #'float8_e5m2fnuz',
        #'float8_e4m3fn',
        #'float8_e4m3fnuz',
        ]

stats = ["max adiff", "max rdiff", "num adiff", "num rdiff"]

iters = 10

if __name__ == "__main__":
    header = {}
    print_lines_min = {}
    print_lines_avg = {}
    print_lines_max = {}
    print_lines_stdev = {}
    for stat in stats:
        header[stat] = "shape,"
        print_lines_min[stat] = []
        print_lines_avg[stat] = []
        print_lines_max[stat] = []
        print_lines_stdev[stat] = []
        for dtype in datatypes:
            header[stat] += "%s," % (dtype)

    #
    # for each gemm size, identify:
    #   1) the maximum absolute abft error
    #   2) the maximum relative abft error
    #   3) the number of elements that exceed atol absolute error
    #   4) the number of elements that exceed rtol absolute error
    # over all output elements in the checksum output vectors.
    #
    # Then calculate the average, min, max, and stdev of these values over
    # <iters> iterations of the abft execution.
    #
    for size in gemm_sizes:
        m = n = k = size
        print_line_min = {}
        print_line_avg = {}
        print_line_max = {}
        print_line_stdev = {}
        for stat in stats:
            print_line_min[stat] = "%dx%dx%d," % (m, n, k)
            print_line_avg[stat] = "%dx%dx%d," % (m, n, k)
            print_line_max[stat] = "%dx%dx%d," % (m, n, k)
            print_line_stdev[stat] = "%dx%dx%d," % (m, n, k)
        for dtype in datatypes:
            max_diffs = []
            max_rdiffs = []
            num_diffs = []
            num_rdiffs = []
            for iter in range(iters):
                (max_diff, max_rdiff, num_diff, num_rdiff) = test_matrix_mul(m,n,k, dtype)
                max_diffs.append(max_diff)
                max_rdiffs.append(max_rdiff)
                num_diffs.append(num_diff)
                num_rdiffs.append(num_rdiff)
                #print("%dx%dx%d %s max diff:%f, max rdiff:%f, num diff:%d, num rdiff:%d" % (m, n, k, dtype, max_diff, max_rdiff, num_diff, num_rdiff))
            print_line_min["max adiff"] += "%f, " % min(max_diffs)
            print_line_min["max rdiff"] += "%f, " % min(max_rdiffs)
            print_line_min["num adiff"] += "%d, " % min(num_diffs)
            print_line_min["num rdiff"] += "%d, " % min(num_rdiffs)

            print_line_avg["max adiff"] += "%f, " % (sum(max_diffs)/len(max_diffs))
            print_line_avg["max rdiff"] += "%f, " % (sum(max_rdiffs)/len(max_rdiffs))
            print_line_avg["num adiff"] += "%f, " % (sum(num_diffs)/len(num_diffs))
            print_line_avg["num rdiff"] += "%f, " % (sum(num_rdiffs)/len(num_rdiffs))

            print_line_max["max adiff"] += "%f, " % max(max_diffs)
            print_line_max["max rdiff"] += "%f, " % max(max_rdiffs)
            print_line_max["num adiff"] += "%d, " % max(num_diffs)
            print_line_max["num rdiff"] += "%d, " % max(num_rdiffs)

            if len(max_diffs) > 1:
                print_line_stdev["max adiff"] += "%f, " % statistics.stdev(max_diffs)
                print_line_stdev["max rdiff"] += "%f, " % statistics.stdev(max_rdiffs)
                print_line_stdev["num adiff"] += "%f, " % statistics.stdev(num_diffs)
                print_line_stdev["num rdiff"] += "%f, " % statistics.stdev(num_rdiffs)

        for stat in stats:
            print_lines_min[stat].append(print_line_min[stat])
            print_lines_avg[stat].append(print_line_avg[stat])
            print_lines_max[stat].append(print_line_max[stat])
            print_lines_stdev[stat].append(print_line_stdev[stat])
    for stat in stats:
        print("\n")
        print("#### %s (avg) ####" % stat)
        print(header[stat])
        for print_line in print_lines_avg[stat]:
            print(print_line)
        if iters > 1:
            print("#### %s (min) ####" % stat)
            print(header[stat])
            for print_line in print_lines_min[stat]:
                print(print_line)
            print("#### %s (max) ####" % stat)
            print(header[stat])
            for print_line in print_lines_max[stat]:
                print(print_line)
            print("#### %s (stdev) ####" % stat)
            print(header[stat])
            for print_line in print_lines_stdev[stat]:
                print(print_line)
    print("\nTest complete")

