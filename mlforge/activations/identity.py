from mlforge.core.module import Module


class Identity(Module):

    def forward(self, x):
        return x