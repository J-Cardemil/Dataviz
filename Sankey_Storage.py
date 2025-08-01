import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

fig = plt.figure(figsize=(8, 4), dpi=80)
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])

for spine in ax.spines.values():
    spine.set_visible(False)

flows = [100, -0.5, -3, -9.6, -0.5, -86]
sankey = Sankey(ax=ax, scale=0.015, offset=0.4, head_angle=180, format='%.1f', unit='%')
sankey.add(flows=flows,label='Carga',
           orientations=[0, 1, 1, 1, 1,-1],
           pathlengths=[1.3, 0.5, 1, 1.5, 2, 0.8],
           rotation=90,
           facecolor='tab:blue')
sankey.add(flows=[86, -0.8, -6.5, -1.4, -0.5, -77.3], label='Descarga',
           orientations=[-1,1, 1, 1, 1,0 ], prior=0, connect=(5, 0),
           pathlengths=[0.4, 0.5, 1, 1.5, 2, 0.1],
           facecolor='tab:green')
diagrams = sankey.finish()
diagrams[-1].patch.set_hatch('/')
plt.legend(loc='lower left')

# %%
# Notice that only one connection is specified, but the systems form a
# circuit since: (1) the lengths of the paths are justified and (2) the
# orientation and ordering of the flows is mirrored.
plt.savefig('transparent_plot.png', transparent=True)

plt.show()