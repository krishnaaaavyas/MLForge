from mlforge.core.parameter import Parameter


p = Parameter(5)

print(p)

p.grad = 3

print(p)

p.zero_grad()

print(p)