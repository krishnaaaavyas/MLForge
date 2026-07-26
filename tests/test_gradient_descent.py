from mlfs.optimizers import GradientDescent

optimizer = GradientDescent(learning_rate=0.2)

print(optimizer.step(weight=10, gradient=15))