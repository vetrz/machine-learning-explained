import math


class Value(object):
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.grad = 0.0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), (
            "only supporting int/float powers for now"
        )

        out = Value(data=self.data**other, _children=(self,), _op=f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)

        out = Value(
            data=t,
            _children=(self,),
            _op="tanh",
        )

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(data=math.exp(x), _children=(self,), _op="exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other):
        out = self + other

        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        out = self + (-other)
        out._op = "-"

        return out

    def __rsub__(self, other):
        out = other + (-self)
        out._op = "-"

        return out

    def __truediv__(self, other):
        out = self * other**-1
        out._op = "/"

        return out

    def __rtruediv__(self, other):
        out = other * self**-1
        out._op = "/"

        return out

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"