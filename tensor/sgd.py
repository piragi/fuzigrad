class SGD():
    def __init__(self, start, lr=0.01):
        self.start = start
        self.lr = lr

    def optimize(self):
        visited = set()
        leafs = []

        def dfs(v):
            if v not in visited:
                visited.add(v)
                if v._prev:
                    for prev in v._prev:
                        dfs(prev)
                else:
                    leafs.append(v)
        dfs(self.start)

        for v in leafs:
            v.value -= self.lr * v.grad.value