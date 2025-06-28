import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib.animation import PillowWriter

# Load Excel data
df = pd.read_excel("costos-marginales-online.xlsx", sheet_name='amCharts')
df['item'] = pd.to_datetime(df['item'])

# Prepare x and y
x = df['item']
y = df['Crucero']

# Load background image
bg_img = mpimg.imread("Image.png")
x_num = mdates.date2num(x)
x_min, x_max = x.min(), x.max()
x_min_num, x_max_num = mdates.date2num([x_min, x_max])
y_min, y_max = -20, y.max()

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor('none')  # transparent plot area

ax.imshow(bg_img, extent=[x_min_num, x_max_num, y_min, y_max], aspect='auto', zorder=0)

ax.set_title("Costo Marginal Horario - Nudo Crucero",  fontsize=20)
ax.set_xlabel("Hora del Día",  fontsize=16)
ax.set_ylabel("Costo Marginal ($/MWh)",  fontsize=16)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(-5, y.max() + 10)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Empty line to update
line, = ax.plot([], [], color='orange', linewidth=2)


# Define shaded intervals
buy_start = pd.Timestamp("2024-12-05 9:00:00")
buy_end = pd.Timestamp("2024-12-05 19:00:00")
sell_start = pd.Timestamp("2024-12-05 21:00:00")
sell_end = pd.Timestamp("2024-12-05 23:00:00")

# Init function
def init():
    line.set_data([], [])
    return line,

# Update function for animation
def update(frame):
    if frame <= len(x):
        line.set_data(x[:frame], y[:frame])
    if frame == len(x):
        # Add "Buy" shaded area
        ax.axvspan(buy_start, buy_end, color='gray', alpha=0.3)
        ax.text(buy_start + pd.Timedelta(hours=5), y.max() * 0.9, "Buy",
                color='black', fontsize=12, ha='center')

        # Add "Sell" shaded area
        ax.axvspan(sell_start, sell_end, color='green', alpha=0.3)
        ax.text(sell_start + pd.Timedelta(hours=1), y.max() * 0.9, "Sell",
                color='black', fontsize=12, ha='center')
    return line,

# Build animation
ani = animation.FuncAnimation(
    fig, update, init_func=init,
    frames=len(x) + 10, interval=200, blit=True
)

# Show the animation
plt.tight_layout()
plt.show()

# Optional: Save as GIF
ani.save("costo_marginal_animado.gif", writer=PillowWriter(fps=2))

