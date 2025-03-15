from z3 import *

# Declare variables
x, y, z = Reals('x y z')
solver = Solver()

# Add cube constraints
solver.add(x >= 0.0, x <= 1.0)
solver.add(y >= 0.0, y <= 1.0)
solver.add(z >= 0.0, z <= 1.0)

# Restrict precision to 5 digits
scale = 10  # 10^5
solver.add(x * scale == ToInt(x * scale))
solver.add(y * scale == ToInt(y * scale))
solver.add(z * scale == ToInt(z * scale))

count = 100000

while count > 0:
    # Solve
    if solver.check() == sat:
        model = solver.model()
        print("x:", model[x])
        print("y:", model[y])
        print("z:", model[z])
        solver.add(Or(x != model[x], y != model[y], z != model[z]))
    else:
        print("No solution")
        break

    count -= 1