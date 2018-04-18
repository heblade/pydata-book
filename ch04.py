import numpy as np
import random
from numpy.matlib import randn
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr, eig

np.set_printoptions(suppress=True)

def practicenumpy():
    # data = np.random.randn(2, 3)
    # print("data: \n",data)
    # print("data*10: \n", data * 10)
    # print("data+data: \n", data + data)
    # print("data shape: \n", data.shape)
    # print("data type: \n", data.dtype)

    # data1 = [6, 7.5, 8, 0, 1]
    # arr1 = np.array(data1)
    # print(arr1)
    # print("data shape: \n", arr1.shape)
    # print("data type: \n", arr1.dtype)
    #
    # data2 = [[1,2,3,4],[5,6,7,8]]
    # arr2 = np.array(data2)
    # print(arr2)
    # print("data2 dimension: \n", arr2.ndim)
    # print("data2 shape: \n", arr2.shape)
    # print("data2 type: \n", arr2.dtype)
    # arrzero = np.zeros(10)
    # print(arrzero)
    #
    # arrzero2 = np.zeros((3, 6))
    # print(arrzero2)
    #
    # arrempty = np.empty((2, 3, 2))
    # print(arrempty)
    # print(arrempty.ndim, arrempty.shape, arrempty.dtype)
    #
    # nprange = np.arange(15)
    # print(nprange)
    #
    # npeye = np.eye(10)
    # print("npeye: \n", npeye)
    # npeye = npeye.astype(np.int32)
    # print(npeye)
    # numberic_strings = np.array(["1.25", "-9.6", "42"], dtype= np.string_)
    # print(numberic_strings)
    # numberic_strings = numberic_strings.astype(float)
    # print(numberic_strings)
    # int_array = np.arange(10)
    # print(int_array)
    # calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
    # int_array = int_array.astype(calibers.dtype)
    # print(int_array)
    #
    # empty_unit32 = np.empty(8, dtype="u4")
    # print(empty_unit32)
    # arr = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    # #点乘
    # print(arr)
    # print(arr * arr)
    # #自身矩阵乘积
    # print(arr.dot(arr))
    # print(1/arr)
    # arr3d = np.array([[[1,2,3], [4,5,6]], [[7,8,9],[10,11,12]]])
    # print(arr3d)
    # print(arr3d[0])
    #
    # old_values = arr3d[0].copy()
    # arr3d[0]=42
    # print(arr3d)
    # arr3d[0] = old_values
    # print(arr3d)
    # print(arr3d[1,0])

    # arr2d = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    # print(arr2d[:2])
    # print(arr2d[:2, 1:])
    # print(arr2d[1, :2])
    # print(arr2d[2, :1])
    # print(arr2d[:,:1])
    # arr2d[:2, 1:] = 0
    # print(arr2d)

    # names = np.array(["Bob", "Joe", "Will", "Bob", "Will","Joe", "Joe"])
    # data = randn(7,4)
    # print(names=="Bob")
    # print(data)
    # print(data[names=="Bob"])
    # print(data[names=="Bob", 2:])
    # print(data[names == "Bob", 3])
    # print(data[~(names=="Bob")])

    # mark = (names == "Bob") | (names == "Will")
    # print(mark)
    # print(data[mark])
    # data[data < 0] = 0
    # print(data)
    #
    # data[names != "Joe"] = 7
    # print(data)
    # arr = np.empty((8,4))
    # for i in range(8):
    #     arr[i] = i
    # print(arr)
    # print(arr[[4,3,0,6]])
    # print(arr[[-3, -5, -7]])
    # arr = np.arange(32).reshape(8,4)
    # print(arr)
    # print(arr[[1,5,7,2],[0,3,1,2]])
    # print(arr[[1,5,7,2]][:,[0,3,1,2]])
    # print(arr[np.ix_([1,5,7,2],[0,3,1,2])])
    # arr = np.arange(15).reshape(3, 5)
    # print(arr)
    # print(arr.T)
    # arr = np.random.randn(6,3)
    # print(arr)
    # print(np.dot(arr.T, arr))
    # arr = np.arange(16).reshape(2,2,4)
    # print(arr)
    # print(arr.transpose(1,0,2))
    # print(arr.swapaxes(0,2))

    # arr = np.arange(10)
    # print(np.sqrt(arr))
    # print(np.exp(arr))
    #
    # x = randn(8)
    # y = randn(8)
    # print(x)
    # print(y)
    # print(np.maximum(x, y))
    # arr = randn(7) * 5
    # print(arr)
    # modfout = np.modf(arr)
    # print(modfout)
    # #print(modfout.shape)
    # print(modfout[0][:,:3])

    # points = np.arange(-5, 5, 0.01)
    # #print(points)
    # xs, ys = np.meshgrid(points, points)
    # #print(xs[:10])
    # #print(ys[:10])
    # z = np.sqrt(xs**2+ys**2)
    # # print(z[:10])
    # plt.imshow(z, cmap=plt.cm.gray)
    # plt.colorbar()
    # plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
    # plt.show()

    # xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    # yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
    # cond = np.array([True, False, True, True, False])
    # result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
    # print(result)
    #
    # arr = randn(4,4)
    # print(arr)
    # print(np.where(arr > 0.2, 2, -2))
    # print(np.where(arr>0.2, 2, arr))
    # arr = np.random.randn(5, 4)
    # print(arr)
    # print(arr.mean())
    # print(arr.sum())
    # print(arr.mean(axis=1))
    # print(arr.sum(0))
    # arr = np.array([[0,1,2],[3,4,5],[6,7,8]])
    # print(arr)
    # print(arr.cumsum())
    # print(arr.cumsum(0))
    # print(arr.cumprod())
    # print(arr.cumprod(0))
    # arr=randn(100)
    # print(arr)
    # print((arr>0).sum())
    # bools = np.array([False, False, True, False])
    # print(bools.any())
    # print(bools.all())
    # arr = randn(8)
    # print(arr)
    # arr.sort()
    # print(arr)

    # arr = randn(5, 3)
    # print(arr)
    # # arr.sort()
    # # print(arr)
    # #每列从小到大排序
    # arr.sort(0)
    # print(arr)
    # #每行从小到大排序
    # arr.sort(1)
    # print(arr)

    # large_arr = np.random.randn(1000)
    # large_arr.sort()
    # print(large_arr)
    # print(large_arr[int(0.05*len(large_arr))])

    # names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
    # print(np.unique(names))
    # print(sorted(set(names)))
    # ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
    # print(np.unique(ints))

    # values = np.array([6, 0, 0, 3, 2, 5, 6])
    # print(np.in1d(values, [2, 3, 6]))

    # arr = np.arange(10)
    # np.save("some_arry", arr)
    # arr = np.load("some_arry.npy")
    # print(arr)
    # arr = np.arange(10)
    # np.savez("array_archive.npz", a=arr, b=arr)
    # arch = np.load("array_archive.npz")
    # print(arch["b"])
    # print(arch["a"])
    arr = np.loadtxt("./datasets/movielens/ratings.dat", delimiter="::")
    print(arr[:10])

def linearalgebra():
    # x = np.array([[1., 2., 3.], [4., 5., 6.]])
    # y = np.array([[6., 23.], [-1, 7], [8, 9]])
    # print("x is: \n",x)
    # print("y is: \n", y)
    # print("x*y is \n",x.dot(y)) #equal with np.dot(x, y)
    # print("x*1 is\n", np.dot(x, np.ones(3)))

    X = np.random.rand(5, 5)
    print("X is: \n", X)
    print("X.T is: \n", X.T)
    mat = X.T.dot(X)
    print("X.T*X is: \n", mat)
    matinverse = inv(mat)
    print("Inverse of X.T*X is: \n", matinverse)
    print(mat.dot(matinverse))
    q, r = qr(mat)
    print("Q of QR is: \n", q)
    print("R of QR is: \n", r)

    x = np.array([[1., 2., 3.], [4., 5., 6.], [7, 8, 9]])
    w, sv = eig(x)
    print("Eigenvalue is: \n", w)
    print("Eigenvector is: \n", sv)

def nprandom():
    samples = np.random.normal(size=(4,4))
    print(samples)

def randomwalk():
    position = 0
    walk = [position]
    steps = 1000
    for i in range(steps)[0:999]:
        step = 1 if random.randint(0, 1) else -1
        position += step
        walk.append(position)
    print(walk)
    plt.plot(range(steps), walk, "b-")
    plt.xlabel("Steps")
    plt.ylabel("Walk")
    plt.title("Random walk with +1/-1 steps")
    plt.show()

def multirandomwalk():
    nwalks = 5
    nsteps = 1000
    draws = np.random.randint(0, 2, size=(nwalks, nsteps))
    steps = np.where(draws > 0, 1, -1)
    walks = steps.cumsum(1)
    print(walks.max())
    print(walks.min())

    hits30 = (np.abs(walks)>=30).any(1)
    print(hits30.sum())
    color = ["r-", "b-", "g-", "y-", "k-"]
    for i in range(nwalks):
        plt.plot(range(nsteps), walks[i,:], color[i], label = ("walk_%d" % i))
    plt.xlabel("Steps")
    plt.ylabel("Walk")
    plt.title("Random walk with +1/-1 steps")
    plt.show()

def throwcoin():
    nsteps = 1000
    draws = np.random.randint(0, 2, size=nsteps)
    steps = np.where(draws > 0, 1, -1)
    print(steps)
    walk = steps.cumsum()
    print(walk)
    print(walk.min())
    print(walk.max())
    # plt.plot(range(nsteps), walk, "b-")
    # plt.xlabel("Steps")
    # plt.ylabel("Record")
    # plt.title("Throw Coin Cumsum")
    # plt.show()


if __name__ == "__main__":
    #practicenumpy()
    #linearalgebra()
    # nprandom()
    # randomwalk()
    #throwcoin()
    multirandomwalk()