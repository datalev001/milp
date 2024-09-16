import pulp
import pandas as pd
import numpy as np

# Parameters
num_days = 30  # Number of days in a month
num_agents = 5  # Number of agents
shifts = ['E', 'F']  # Early shift 'E' and late shift 'F'
max_consecutive_days = 5  # Increased consecutive days limit for easier constraint handling
min_shifts_per_day = 1  # Minimum number of shifts per day

# Create the problem
prob = pulp.LpProblem("SchedulingProblem", pulp.LpMinimize)

# Decision variables: x[i, d, s] is 1 if agent i is assigned to shift s on day d, 0 otherwise
x = pulp.LpVariable.dicts("x", ((i, d, s) for i in range(num_agents) for d in range(num_days) for s in shifts), cat='Binary')

# Workload variance variable
workload_variance = pulp.LpVariable("workload_variance", lowBound=0, cat='Continuous')

# Total shifts worked by each agent
workloads = [pulp.lpSum([x[i, d, s] for d in range(num_days) for s in shifts]) for i in range(num_agents)]

# Objective: Minimize the workload variance
prob += workload_variance

# Constraints: Each agent can only work one shift per day
for i in range(num_agents):
    for d in range(num_days):
        prob += pulp.lpSum([x[i, d, s] for s in shifts]) <= 1, f"One_shift_per_day_{i}_{d}"

# Constraint: At least one agent must be assigned to each shift per day
for d in range(num_days):
    for s in shifts:
        prob += pulp.lpSum([x[i, d, s] for i in range(num_agents)]) >= min_shifts_per_day, f"Min_shifts_day_{d}_{s}"

# Constraint: No agent works more than max_consecutive_days in a row
for i in range(num_agents):
    for d in range(num_days - max_consecutive_days + 1):
        prob += pulp.lpSum([x[i, d + k, s] for k in range(max_consecutive_days) for s in shifts]) <= max_consecutive_days, f"Max_consecutive_days_{i}_{d}"

# Balance workload variance
for i in range(num_agents):
    prob += workloads[i] - workload_variance <= 0
    prob += workload_variance - workloads[i] <= 0

# Solve the problem
prob.solve()

# Output the schedule
for i in range(num_agents):
    print(f"Agent {i+1}'s schedule:")
    for d in range(num_days):
        for s in shifts:
            if pulp.value(x[i, d, s]) == 1:
                print(f"  Day {d+1}: Shift {s}")

# Output the minimized workload variance
print(f"Minimized workload variance: {pulp.value(workload_variance)}")

# Validation
errors = []
for i in range(num_agents):
    print(f"\nValidating Agent {i+1}'s schedule...")  # Intermediate result for each agent
    
    # Check for more than one shift per day
    for d in range(num_days):
        shifts_worked = [pulp.value(x[i, d, s]) for s in shifts]
        print(f"  Day {d+1}: Shifts worked = {shifts_worked}")  # Print the shifts worked on each day
        if sum(shifts_worked) > 1:
            errors.append(f"Agent {i+1} has more than one shift on Day {d+1}")
    
    # Check for consecutive days constraint
    for d in range(num_days - max_consecutive_days + 1):
        consecutive_shifts = [pulp.value(x[i, d + k, s]) for k in range(max_consecutive_days) for s in shifts]
        print(f"  Days {d+1} to {d+max_consecutive_days}: Consecutive shifts worked = {sum(consecutive_shifts)}")  # Print consecutive shifts
        if sum(consecutive_shifts) > max_consecutive_days:
            errors.append(f"Agent {i+1} works more than {max_consecutive_days} consecutive days starting on Day {d+1}")

if errors:
    print("\nModel validation failed with the following errors:")
    for error in errors:
        print(error)
else:
    print("\nModel validation passed!")
