import pandas as pd
import plotly.graph_objects as go

# Cargar los datos
df = pd.read_excel("costos-marginales-online.xlsx", sheet_name='amCharts')
df['item'] = pd.to_datetime(df['item'])
df = df.rename(columns={'item': 'Hora', 'Crucero': 'Costo_Marginal'})
df = df.sort_values(by='Hora').reset_index(drop=True)

# Datos
x = df['Hora']
y = df['Costo_Marginal']

# Definir rangos para las zonas Buy y Sell
buy_start = pd.Timestamp("2024-12-05 09:00:00")
buy_end = pd.Timestamp("2024-12-05 19:00:00")
sell_start = pd.Timestamp("2024-12-05 21:00:00")
sell_end = pd.Timestamp("2024-12-05 23:00:00")

# Create base figure (without buy/sell highlights initially)
fig = go.Figure(
    layout=go.Layout(
        title="Animación de Costo Marginal - Nodo Crucero",
        xaxis=dict(title="Hora", range=[x.min(), x.max()]),
        yaxis=dict(title="Costo Marginal ($/MWh)", range=[-10, y.max() + 20]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        updatemenus=[dict(type="buttons", showactive=False,
                          buttons=[dict(label="Play", method="animate", args=[None])])]
    )
)

# Initial trace (just the first point)
fig.add_trace(go.Scatter(
    x=[x[0]], y=[y[0]],
    mode='lines+markers',
    name='Costo Marginal',
    line=dict(color='orange')
))

# Create frames: one per data point, add shapes only in final frame
frames = []
for k in range(1, len(x)):
    shapes = []
    annotations = []

    # If it's the last frame, add the shaded areas and labels
    if k == len(x) - 1:
        shapes = [
            dict(type="rect", x0=buy_start, x1=buy_end, y0=0, y1=y.max()+20,
                 fillcolor="gray", opacity=0.3, layer="below", line_width=0),
            dict(type="rect", x0=sell_start, x1=sell_end, y0=0, y1=y.max()+20,
                 fillcolor="green", opacity=0.2, layer="below", line_width=0)
        ]
        annotations = [
            dict(x=buy_start + (buy_end - buy_start)/2, y=y.max()*0.95,
                 text="Buy", showarrow=False, font=dict(size=14, color="black")),
            dict(x=sell_start + (sell_end - sell_start)/2, y=y.max()*0.95,
                 text="Sell", showarrow=False, font=dict(size=14, color="black"))
        ]

    frames.append(go.Frame(
        data=[go.Scatter(x=x[:k+1], y=y[:k+1], mode='lines+markers',
                         line=dict(color='orange'))],
        layout=go.Layout(shapes=shapes, annotations=annotations)
    ))

fig.frames = frames

# Save updated HTML
html_delayed_zones = "marginal_plotly_animacion_zonas_al_final.html"
fig.write_html(html_delayed_zones)

html_delayed_zones