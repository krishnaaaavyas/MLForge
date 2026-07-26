
from mlforge.core.parameter import Parameter
from mlforge.optimizers import GradientDescent

weight = Parameter(10)

weight.grad = 15

optimizer = GradientDescent(
    learning_rate=0.2
)

optimizer.update(weight)

print(weight.value)

from mlforge.optimizers import GradientDescent

optimizer = GradientDescent(learning_rate=0.2)

print(optimizer.step(weight=10, gradient=15))

