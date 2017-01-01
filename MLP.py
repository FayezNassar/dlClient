from chainer import FunctionSet, Variable, Chain
import chainer.functions as F


# Network definition
class MLP(Chain):
    def __init__(self, ll1, ll2):
        super(MLP, self).__init__(
            l1=ll1,
            l2=ll2,
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        return h2

    def mlp_forward(self, x):
        out1 = self.l1(x)
        out2 = F.sigmoid(out1)
        return F.sigmoid(self.l2(out2))

    def classify(self, x_data):
        x = Variable(x_data)
        h = self.mlp_forward(x)
        return h.data[0].tolist().index(max(h.data[0]))