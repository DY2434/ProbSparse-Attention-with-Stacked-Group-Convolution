import numpy as np
from numba import njit, prange

def generate_kernels(input_length, num_kernels):
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)  #定义了三个可选的卷积核长度候选值:7、9和11
    lengths = np.random.choice(candidate_lengths, num_kernels) #随机选择num_kernels个长度值,存储在lengths数组中

    weights = np.zeros(lengths.sum(), dtype=np.float64)  #存储所有卷积核的权重值,总长度为所有卷积核长度之和
    biases = np.zeros(num_kernels, dtype=np.float64)     #存储每个卷积核的偏置值
    dilations = np.zeros(num_kernels, dtype=np.int32)    #存储每个卷积核的膨胀因子
    paddings = np.zeros(num_kernels, dtype=np.int32)     #存储每个卷积核的填充大小

    a1 = 0

    for i in range(num_kernels):
        _length = lengths[i]                          #获取当前卷积核的长度
        _weights = np.random.normal(0, 1, _length)    #生成一个服从标准正态分布的权重数组，0均值，1标准差，生成的长度
        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean()   #将权重数组减去均值,得到去中心化后的权重,并存储到weights数组中
        biases[i] = np.random.uniform(-1, 1)          #生成一个服从均匀分布的偏置值
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation
        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding
        a1 = b1

    return weights, lengths, biases, dilations, paddings

#使用Numba库的装饰器,可以将Python函数编译成高效的机器码,从而大幅提高运行速度。fastmath=True选项可以进一步优化数学运算。
@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding):
    input_length = len(X)
    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)
    _ppv = 0        #初始化两个输出特征变量：ppv正值百分比，最大值
    _max = np.NINF  #负无穷大
    end = (input_length + padding) - ((length - 1) * dilation)

    for i in range(-padding, end):
        _sum = bias
        index = i
        for j in range(length):
            if index > -1 and index < input_length:    #如果索引在输入序列的有效范围内,则将该位置的输入值乘以对应的卷积核权重,并累加到_sum中。
                _sum = _sum + weights[j] * X[index]
            index = index + dilation
        if _sum > _max:   #更新_max变量,如果当前_sum大于之前的最大值,则更新_max
            _max = _sum
        if _sum > 0:      #如果当前_sum大于0,则_ppv加1
            _ppv += 1

    return _ppv / output_length, _max

@njit("float64[:,:](float64[:,:],Tuple((float64[::1],int32[::1],float64[::1],int32[::1],int32[::1])))", parallel=True, fastmath=True)
def apply_kernels(X, kernels):
    weights, lengths, biases, dilations, paddings = kernels
    num_examples, _ = X.shape
    num_kernels = len(lengths)
    _X = np.zeros((num_examples, 2*num_kernels), dtype=np.float64)  #初始化一个新的特征矩阵_X,形状为(num_examples, 2*num_kernels),用于存储最终的特征。

    for i in prange(num_examples):  #prange并行循环
        a1 = 0   #初始化两个变量a1和a2,用于跟踪当前处理的卷积核在weights、lengths等数组中的位置。
        a2 = 0
        for j in range(num_kernels):
            b1 = a1 + lengths[j]  #计算当前卷积核在weights数组中的起止位置b1
            b2 = a2 + 2
            _X[i, a2:b2] = apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])
            a1 = b1     #更新a1和a2以便于处理下一个卷积核
            a2 = b2

    return _X