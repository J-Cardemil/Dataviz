import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Pumped Hydro Energy Flow")

sankey = Sankey(ax=ax, unit=None, scale=1.0, gap=0.5)

# Storage side
sankey.add(flows=[100, -0.5, -9.6, -3, -0.5, -86.4],
            labels=['Storage Input', 'Tunnel Loss', 'Pump Loss', 'Motor Loss', 'Transformer Loss', 'To Generation'],
            orientations=[0, 1, 1, 1, 1, 0],
            facecolor='cornflowerblue')

# Generation side
sankey.add(flows=[86.4, -0.8, -6.5, -1.4, -0.5, -77.3],
            labels=['From Storage', 'Tunnel Loss', 'Turbine Loss', 'Generator Loss', 'Transformer Loss', 'Generation Output'],
            orientations=[0, -1, -1, -1, -1, 0],
            prior=0, connect=(5, 0),
            facecolor='royalblue')

sankey.finish()
plt.title("Energy Flow in Pumped Hydro Storage and Generation System")
plt.tight_layout()
plt.show()
