import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
from matplotlib.animation import PillowWriter

# Cargar archivo Excel (ajusta el path si es necesario)
df = pd.read_excel("costos-marginales-online.xlsx", sheet_name='amCharts')

# Convertir fechas
df['item'] = pd.to_datetime(df['item'])


# Convertir a formato numérico para matplotlib
x = mdates.date2num(df['item'])
y = df['Crucero']

# %%


# Crear figura
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Costo Marginal Horario - Nudo Crucero (Enero 2025)")
ax.set_xlabel("Hora del Día")
ax.set_ylabel("Costo Marginal ($/MWh)")
ax.set_ylim(-5, y.max() + 10)
ax.set_xlim(x.min(), x.max())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

line, = ax.plot(x, y, '-', color='orange')

plt.show()
# %%

# %%


# Definir el rango a sombrear (21:00 a 24:00)
highlight_start = pd.Timestamp("2024-12-05 21:00:00")
highlight_end = pd.Timestamp("2024-12-05 23:00:00")
highlight_mask = (df['item'] >= highlight_start) & (df['item'] < highlight_end)
x_highlight = x[highlight_mask]
y_highlight = y[highlight_mask]


# Función para animación
def update(frame):
    ax.collections.clear()
    if frame % 20 < 10:
        ax.fill_between(x_highlight, y_highlight, color='red', alpha=0.3)

# Crear animación
ani = animation.FuncAnimation(fig, update, frames=40, interval=200)

# Guardar como GIF
ani.save("arbitraje_animado.gif", writer=PillowWriter(fps=5))