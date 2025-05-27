import matplotlib.pyplot as plt
import numpy as np

# Sample data
classification = ['Combined\n cycle\n gas turbine', 'Simple\n cycle\n gas turbine', 'Diabatic\n CAES', 'Adiabatic\n CAES', 'Li Ion\n battery']# classification = ['Combined\n cycle\n gas turbine','' ,'Simple\n cycle\n gas turbine','' , 'Diabatic\n CAES','' , 'Adiabatic\n CAES','' , 'Li Ion\n battery']
variables = ['O&M cost ($/MWh out)', 'Electricity use ($/MWh out)', 'Fuel use ($/MWh out)', 'Total CAPEX ($/MWh out)']

# Values for each variable per classification
data = {
    'O&M cost ($/MWh out)': [2, 3, 2, 1, 3],
    'Electricity use ($/MWh out)': [0, 0, 2, 7, 6],
    'Fuel use ($/MWh out)': [29, 39, 15, 0, 0],
    'Total CAPEX ($/MWh out)': [20, 22, 61, 83, 82],
}

# Convert to numpy array for easier calculations
values = np.array([data[var] for var in variables])  # shape: (variables, classification)

# Select specific colors from tab20
cmap = plt.get_cmap('tab20')
color_indices = {
    'O&M cost ($/MWh out)': 8,
    'Electricity use ($/MWh out)': 5,
    'Fuel use ($/MWh out)': 6,
    'Total CAPEX ($/MWh out)': 0,
}
colors = {var: cmap(i) for var, i in color_indices.items()}

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Reverse order for stacking from bottom up
reversed_variables = list(reversed(variables))
reversed_values = values[::-1]

# Stack bars
bottom = np.zeros(len(classification))
for i, var in enumerate(reversed_variables):
    ax.bar(classification, reversed_values[i], bottom=bottom, label=var, color=colors[var], zorder=10)
    bottom += reversed_values[i]

# Final touches
ax.set_ylabel('Levelized cost of electricity ($/MWh)')
ax.legend()
ax.grid(axis='y', alpha=0.7, zorder=0)
plt.yticks(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
plt.ylim(0, 100)
plt.tight_layout()
plt.show()

