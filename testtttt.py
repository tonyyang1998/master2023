from gurobipy import Model, GRB, quicksum
import gurobipy as gp

model = Model('RRP')

x = model.addVar(1, 10)
y = model.addVar(1, 10)
z = model.addVar(1, 10)

model.update()
model.ModelSense = GRB.MAXIMIZE

# Primary objective: x + 2 y
ob1 = model.setObjectiveN(x + y, index = 0, priority = 2)
# Alternative, lower priority objectives: 3 y + z and x + z
ob2 = model.setObjectiveN(-3*y + z, index = 1, priority =1)
ob3 = model.setObjectiveN(x + z, index = 2, priority = 0)

model.addConstr(z <= 8.0, "c1")

model.optimize()

if model.SolCount > 0:
  print(x.X)
  print(y.X)
  print(z.X)