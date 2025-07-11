# app.py
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1) CYLINDER & FEATURE DIMENSIONS (all units in mm)
# ---------------------------------------------------------
R_outer = 1134 / 2      # Outer cylinder radius = 567 mm
H_total = 334           # Total height of cylinder = 334 mm

# Concentric feature diameters on the TOP and BOTTOM faces:
r1 = 892.0 / 2      # 446.0 mm (dashed construction circle)
r2 = 699.2 / 2      # 349.6 mm (dashed construction circle)
r3 = 506.6 / 2      # 253.3 mm (dashed construction circle)
r4 = 116.0  / 2     # 58.0 mm  (solid inlet/outlet tube)

# Thermocouple hole radius = 21.5 mm / 2
r_tc = 21.5 / 2     # 10.75 mm

# ---------------------------------------------------------
# 2) THERMOCOUPLE TREE LOCATIONS (XY COORDS ON TOP FACE)
# ---------------------------------------------------------
north_centers = [
    (0.0, +r1),        # (0, +446.0)
    (0.0, +r2),        # (0, +349.6)
    (0.0, +157.0)      # (0, +157)
]

angle_left  = np.deg2rad(210)  # “left‐bottom” at 210°
left_centers = [
    (r2 * np.cos(angle_left), r2 * np.sin(angle_left)),  # (349.6 at 210°)
    (r3 * np.cos(angle_left), r3 * np.sin(angle_left))   # (253.3 at 210°)
]

angle_right = np.deg2rad(330)  # “right‐bottom” at 330°
right_centers = [
    (r3 * np.cos(angle_right), r3 * np.sin(angle_right)),  # (253.3 at 330°)
    (r1 * np.cos(angle_right), r1 * np.sin(angle_right))   # (446.0 at 330°)
]

tree_centers = north_centers + left_centers + right_centers

# ---------------------------------------------------------
# 3) EACH TREE’S TC LABELS (TOP → BOTTOM)
# ---------------------------------------------------------
tree_labels = [
    ["T2",  "T3",  "T4",  "T5"],    # TC_N1
    ["T6",  "T7",  "T8",  "T9"],    # TC_N2
    ["T10", "T11", "T12", "T13"],   # TC_N3
    ["T14", "T15", "T16", "T17"],   # TC_L1
    ["T18", "T19", "T20", "T21"],   # TC_L2
    ["T22", "T23", "T24", "T25"],   # TC_R1
    ["T26", "T27", "T28", "T29"]    # TC_R2
]

# ---------------------------------------------------------
# 4) Z‐LEVELS: bottom‐edge, 4 TC layers, top‐edge, +30mm in/out
# ---------------------------------------------------------
#   z = -30   (bottom of inner tube)
#   z =  0    (bottom shell)
#   z =  69.7 (4th TC layer)
#   z = 135.7 (3rd TC layer)
#   z = 201.7 (2nd TC layer)
#   z = 267.7 (1st TC layer)
#   z =  334  (top shell)
#   z =  364  (top of inner tube)
z_levels = [-30.0, 0.0, 69.7, 135.7, 201.7, 267.7, 334.0, 364.0]

# ---------------------------------------------------------
# 5) BUILD 3D TRACES
# ---------------------------------------------------------
theta = np.linspace(0, 2 * np.pi, 80)
traces = []

# 5a) OUTER CYLINDER WIREFRAME (vertical spokes + top/bottom rings)
for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
    x_line = R_outer * np.cos(ang) * np.ones(2)
    y_line = R_outer * np.sin(ang) * np.ones(2)
    z_line = np.array([z_levels[1], z_levels[6]])  # 0 → 334
    traces.append(
        go.Scatter3d(
            x=x_line, y=y_line, z=z_line,
            mode='lines', line=dict(color='gray', width=1)
        )
    )

# Bottom shell ring (z = 0)
x_circ = R_outer * np.cos(theta)
y_circ = R_outer * np.sin(theta)
traces.append(
    go.Scatter3d(
        x=x_circ, y=y_circ, z=np.full_like(theta, z_levels[1]),
        mode='lines', line=dict(color='gray', width=1)
    )
)
# Top shell ring (z = 334)
traces.append(
    go.Scatter3d(
        x=x_circ, y=y_circ, z=np.full_like(theta, z_levels[6]),
        mode='lines', line=dict(color='gray', width=1)
    )
)

# 5b) CONCENTRIC CIRCLES ON BOTH TOP (z=334) AND BOTTOM (z=0) FACES
for r, dash_style, zface in [
    (r1, 'dash', z_levels[1]),   # bottom = 0
    (r2, 'dash', z_levels[1]),
    (r3, 'dash', z_levels[1]),
    (r4, 'solid', z_levels[1]),
    (r1, 'dash', z_levels[6]),   # top = 334
    (r2, 'dash', z_levels[6]),
    (r3, 'dash', z_levels[6]),
    (r4, 'solid', z_levels[6])
]:
    style = dict(color='black', width=2)
    if dash_style == 'dash':
        style['dash'] = 'dash'
    traces.append(
        go.Scatter3d(
            x=r * np.cos(theta),
            y=r * np.sin(theta),
            z=np.full_like(theta, zface),
            mode='lines',
            line=style
        )
    )

# 5c) INNER “INLET/OUTLET” TUBE CIRCLES ONLY (remove vertical stems)
#    Two circles: z = -30 (bottom of tube) and z = +364 (top of tube)
traces.append(
    go.Scatter3d(
        x=r4 * np.cos(theta),
        y=r4 * np.sin(theta),
        z=np.full_like(theta, z_levels[0]),  # z = -30
        mode='lines', line=dict(color='black', width=2)
    )
)
traces.append(
    go.Scatter3d(
        x=r4 * np.cos(theta),
        y=r4 * np.sin(theta),
        z=np.full_like(theta, z_levels[7]),  # z = 364
        mode='lines', line=dict(color='black', width=2)
    )
)

# 5d) THERMAL COUPLE TREES (7 total): stems from z=69.7 → 267.7 only
for (x0, y0), labels in zip(tree_centers, tree_labels):
    # (i) Vertical red stem (only between the TC layers)
    traces.append(
        go.Scatter3d(
            x=[x0, x0],
            y=[y0, y0],
            z=[z_levels[2], z_levels[5]],  # 69.7 → 267.7
            mode='lines', line=dict(color='red', width=3)
        )
    )
    # (ii) Four red circles + text on each of the four TC layers
    for zi, lbl in zip(z_levels[5:1:-1], labels):
        # zi iterates [267.7, 201.7, 135.7, 69.7] top→bottom
        traces.append(
            go.Scatter3d(
                x=x0 + r_tc * np.cos(theta),
                y=y0 + r_tc * np.sin(theta),
                z=np.full_like(theta, zi),
                mode='lines',
                line=dict(color='red', width=2)
            )
        )
        traces.append(
            go.Scatter3d(
                x=[x0], y=[y0], z=[zi + 3.0],  # label 3 mm above circle
                mode='text',
                text=[lbl],
                textposition='bottom center',
                textfont=dict(size=10, color='black')
            )
        )

# 5e) THERMOCOUPLE T1 at the very top of the inner tube (z = 364)
traces.append(
    go.Scatter3d(
        x=[0], y=[0], z=[z_levels[7] + 3.0],  # z = 364 + 3 mm
        mode='text',
        text=["T1"],
        textposition='bottom center',
        textfont=dict(size=12, color='red')
    )
)

# 5f) THERMOCOUPLE T30 at the very bottom of the inner tube (z = -30)
traces.append(
    go.Scatter3d(
        x=[0], y=[0], z=[z_levels[0] - 3.0],  # z = -30 - 3 mm
        mode='text',
        text=["T30"],
        textposition='top center',
        textfont=dict(size=12, color='red')
    )
)

# ---------------------------------------------------------
# 6) CAMERA & LAYOUT (Isometric, axes hidden)
# ---------------------------------------------------------
camera = dict(eye=dict(x=1.2, y=1.2, z=0.9))

layout = go.Layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=H_total / (2 * R_outer))
    ),
    scene_camera=camera,
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False
)

fig = go.Figure(data=traces, layout=layout)

# ---------------------------------------------------------
# 7) DASH APP
# ---------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H4("TES Cylinder + Inlet/Outlet Tube (No Inner Stems)"),
    dcc.Graph(
        id="tes-3d",
        figure=fig,
        config={"scrollZoom": True}  # Enable mouse-wheel zoom
    )
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True)
