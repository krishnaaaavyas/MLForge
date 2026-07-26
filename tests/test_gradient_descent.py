from mlforge.core.parameter import Parameter
from mlforge.optimizers import GradientDescent

weight = Parameter(10)

weight.grad = 15

optimizer = GradientDescent(
    learning_rate=0.2
)

optimizer.update(weight)

print(weight.value)