from functools import reduce
#向量计算的功能函数
class VectorOp(object):
    """
    实现向量计算操作
    """
    @staticmethod
    def dot(x, y):
        """
        计算两个向量x和y的内积
        """
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]按元素相乘
        # 变成[x1*y1, x2*y2, x3*y3]
        # 然后利用reduce求和
        return reduce(lambda a, b: a + b, VectorOp.element_multiply(x, y), 0.0)

    @staticmethod
    def element_multiply(x, y):
        """
        将两个向量x和y按元素相乘
        """
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]打包在一起
        # 变成[(x1,y1),(x2,y2),(x3,y3),...]
        # 然后利用map函数计算[x1*y1, x2*y2, x3*y3]
        return list(map(lambda x_y: x_y[0] * x_y[1], zip(x, y)))

    @staticmethod
    def element_add(x, y):
        """
        将两个向量x和y按元素相加
        """
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]打包在一起
        # 变成[(x1,y1),(x2,y2),(x3,y3),...]
        # 然后利用map函数计算[x1+y1, x2+y2, x3+y3]
        return list(map(lambda x_y: x_y[0] + x_y[1], zip(x, y)))

    @staticmethod
    def scala_multiply(v, s):
        """
        将向量v中的每个元素和标量s相乘
        """
        return map(lambda e: e * s, v)

#激活函数
def f(x):
    if x > 0:
        return 1
    else:
        return 0
def model(input, weight, bias):
    out = VectorOp.dot(input, weight) + bias
    return f(out)

def update_weights(input_vec, output, label, rate, weights, bias):
    """
    按照感知器规则更新权重
    """
    # 首先计算本次更新的delta
    # 然后把input_vec[x1,x2,x3,...]向量中的每个值乘上delta，得到每个权重更新
    # 最后再把权重更新按元素加到原先的weights[w1,w2,w3,...]上
    delta = label - output
    print("loss:", delta)
    weights = VectorOp.element_add(
        weights, VectorOp.scala_multiply(input_vec, rate * delta))
    # 更新bias
    bias += rate * delta
    return weights, bias

def train(input_vec, label, rate=0.1, epoch=10):
    weights, bias = [0.0, 0.0], 0
    for _ in range(epoch):
        for j, one_data in enumerate(input_vec):
            output = model(one_data, weights, bias)
            weights, bias = update_weights(one_data, output, label[j], rate, weights, bias)
            # print(weights, bias)
    return weights, bias

if __name__ == '__main__':
    # 构建训练数据
    # 输入向量列表
    #input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    #labels = [1, 0, 0, 0]
    input_vecs = [(1,2),(2.5,3),(1.5,1),(5,4.5),(6,3),(2,2),(3,5),(6,4),(5,3)]   
    labels = [-1,-1,-1,1,1,-1,1,1,1]  
    we, b = train(input_vecs, labels)
    print("最终的权重：",we, b)


    #预测
    input_vec = [0, 0]
    output = model(input_vec, we, b)
    print("预测结果：", output)

