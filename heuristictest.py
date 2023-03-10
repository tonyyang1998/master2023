import gurobipy as gp
from gurobipy import GRB
import random


# Set random seed for reproducibility
random.seed(42)

n =  5 # number of cities
dist = [[0]*n for _ in range(n)]  # initialize distance matrix

# Fill in upper triangle with random distances
for i in range(n):
    for j in range(i+1, n):
        dist[i][j] = random.randint(1, 100)
        
# Fill in lower triangle by symmetry
for i in range(1, n):
    for j in range(i):
        dist[i][j] = dist[j][i]

print(dist)


# Create the Gurobi model
m = gp.Model()

# Create decision variables
x = {}
for i in range(n):
    for j in range(n):
        if i != j:
            x[i, j] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}')

# Set objective function
m.setObjective(gp.quicksum(dist[i][j]*x[i,j] for i in range(n) for j in range(n) if i != j), GRB.MINIMIZE)

# Add constraints
# Each city is visited exactly once
for i in range(n):
    m.addConstr(gp.quicksum(x[i,j] for j in range(n) if i != j) == 1, name=f'visit_{i}')
for j in range(n):
    m.addConstr(gp.quicksum(x[i,j] for i in range(n) if i != j) == 1, name=f'leave_{j}')

m.addConstr(gp.quicksum(x[0,j] for j in range(1,n)) == 1, name='start_at_node_0')

# Add subtour elimination constraints
for i in range(n):
    for j in range(n):
        if i != j:
            m.addConstr(x[i,j] + x[j,i] <= 1, name=f'subtour_{i}_{j}')

# Use construction heuristic to generate a feasible solution

m.setParam(gp.GRB.Param.Heuristics, 0.0)
current_city = 0
unvisited_cities = set(range(1, n))
while unvisited_cities:
    nearest_city = min(unvisited_cities, key=lambda city: dist[current_city][city])
    x[current_city, nearest_city].start = 1
    current_city = nearest_city
    unvisited_cities.remove(current_city)
x[current_city, 0].start = 1

# Solve the model
m.optimize()

def sort_path(path):
    sorted_path = [path[0]]
    for i in range(len(path)):
        for edge in path:
            if sorted_path[-1][1] == edge[0]:
                sorted_path.append(edge)
    return sorted_path

# Print the optimal solution
if m.status == GRB.OPTIMAL:
    result = []
    print('Optimal objective value:', m.objVal)
    
    for i in range(n):
        for j in range(n):
            if i!=j:
                if x[i, j].x==1:
                    result.append((i,j))
    
    
    #for v in m.getVars():
        #print(v, v.x)
