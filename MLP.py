import chainer
import chainer.functions as F


# Network definition
class MLP(chainer.Chain):
    def __init__(self, ll1, ll2):
        super(MLP, self).__init__(
            l1=ll1,
            l2=ll2,
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        return h2
