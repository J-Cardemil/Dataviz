import os
import re
import numpy as np
import pandas as pd
import scipy.interpolate
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
from pandas.errors import EmptyDataError
from CoolProp.CoolProp import PropsSI
import plotly.express as px

# -------------------
# Temperature Dashboard (from Temperature.py)
# -------------------
def temperature_dashboard_layout():
    # Use the exact serve_layout() from Temperature.py
    return serve_layout()

# Copy the full serve_layout from Temperature.py
import dash_bootstrap_components as dbc
from dash import dcc, html

def serve_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Img(
                    src="/assets/Logo_Grupo_Solar_UC.png",
                    style={"width": "260px", "margin": "20px auto 10px auto", "display": "block"},
                    alt="Grupo Solar UC Logo"
                )
            ], width=12)
        ]),
        dbc.Row(dbc.Col(
            html.H1("TES Temperature Dashboard", className="text-center text-primary my-4")
        )),
        dbc.Row([
            dbc.Col([
                html.Label("Experiment Temperature (°C):", className="fw-bold"),
                dcc.Dropdown(
                    id="temp-dropdown",
                    options=[{"label": f"{t} °C", "value": str(t)} for t in [140, 180, 220]],
                    value="180",
                    clearable=False
                ),
                html.Br(),
                html.Label("Iteration File(s) for Sensors:", className="fw-bold"),
                dcc.Dropdown(
                    id="iter-dropdown",
                    multi=True,
                    clearable=False
                ),
                html.Hr(),
                html.Label("Sensor Group:", className="fw-bold"),
                dcc.Dropdown(id="group-dropdown", clearable=False),
                html.Br(),
                html.Label("Select Sensors:", className="fw-bold"),
                dcc.Dropdown(id="sensor-dropdown", multi=True),
                html.Div([
                    html.Button(
                        "Select All",
                        id="select-all-btn",
                        n_clicks=0,
                        className="btn btn-outline-primary me-2"
                    ),
                    html.Button(
                        "Clear All",
                        id="clear-all-btn",
                        n_clicks=0,
                        className="btn btn-outline-danger"
                    )
                ], className="mt-2")
            ], width=4),
            dbc.Col(
                dcc.Graph(
                    id="sensor-graph",
                    style={"height": "40vh"}
                ),
                width=8
            )
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.Label("Iteration for Cylinder (3D):", className="fw-bold"),
                dcc.Dropdown(
                    id="cyl-iter-dropdown",
                    clearable=False
                ),
                html.Br(),
                html.Label("Time Index for Cylinder (S):", className="fw-bold"),
                html.Div([
                    dcc.Slider(
                        id="time-slider",
                        min=0,
                        max=0,
                        step=1,
                        value=0,
                        marks={},
                        updatemode="drag"
                    ),
                    dcc.Input(
                        id="time-index-input",
                        type="number",
                        min=0,
                        value=0,
                        style={"width": "18%", "marginLeft": "2%", "display": "inline-block"}
                    )
                ]),
                html.Br(),
                html.Label("Rings to Display:", className="fw-bold"),
                dcc.Checklist(
                    id="ring-select",
                    options=[
                        {"label": "446 mm",   "value": 446.0},
                        {"label": "349.6 mm", "value": 349.6},
                        {"label": "253.3 mm", "value": 253.3},
                        {"label": "157 mm",   "value": 157.0}
                    ],
                    value=[446.0, 349.6, 253.3, 157.0],
                    labelStyle={"display": "block"}
                )
            ], width=4)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="cylinder-3d"), width=12)
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(
                    "© 2025 Ingeniería UC - Grupo Solar. Todos los derechos reservados.",
                    style={"textAlign": "center", "color": "#888", "marginTop": "40px", "marginBottom": "10px", "fontSize": "1.1em"}
                )
            ], width=12)
        ])
    ], fluid=True)

# -------------------
# Energy Dashboard (from Energy.py)
# -------------------
def energy_dashboard_layout(df_all_energy):
    # Add logo and copyright info to the energy dashboard
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Img(
                    src="/assets/Logo_Grupo_Solar_UC.png",
                    style={"width": "260px", "margin": "20px auto 10px auto", "display": "block"},
                    alt="Grupo Solar UC Logo"
                )
            ], width=12)
        ]),
        dbc.Row(dbc.Col(
            html.H1("TES Energy Dashboard", className="text-center text-primary my-4")
        )),
        dbc.Row([
            dbc.Col([
                html.Label("Temperature (°C)"),
                dcc.Dropdown(
                    id='energy-temp-dropdown',
                    options=[{'label': f"{t}°C", 'value': t}
                             for t in sorted(df_all_energy['Temperature'].unique())],
                    value=sorted(df_all_energy['Temperature'].unique())[0],
                    clearable=False
                ),
            ], style={'width':'200px','display':'inline-block','marginRight':'20px'}),
            dbc.Col([
                html.Label("Iteration(s)"),
                dcc.Dropdown(id='energy-iteration-dropdown', multi=True)
            ], style={'width':'300px','display':'inline-block'}),
        ]),
        dcc.Graph(id='energy-mass-flow-graph'),
        dcc.Graph(id='energy-deltaT-graph'),
        dcc.Graph(id='energy-energy-graph'),
        dcc.Graph(id='energy-stored-energy-graph'),
        dcc.Graph(id='energy-fan-work-graph'),
        dcc.Graph(id='energy-efficiency-graph'),
        dbc.Row([
            dbc.Col([
                html.Div(
                    "© 2025 Ingeniería UC - Grupo Solar. Todos los derechos reservados.",
                    style={"textAlign": "center", "color": "#888", "marginTop": "40px", "marginBottom": "10px", "fontSize": "1.1em"}
                )
            ], width=12)
        ])
    ], fluid=True)

# -------------------
# Combined App
# -------------------
DASHBOARD_OPTIONS = [
    {"label": "Temperature Dashboard", "value": "temperature"},
    {"label": "Energy Dashboard", "value": "energy"}
]

# Prepare df_all_energy for energy dashboard
BASE_DIR    = os.path.dirname(__file__)
DATA_PARENT = os.path.join(BASE_DIR, 'data')
iter_re     = re.compile(r'^.*_(\d+)\.csv$')
frames      = []
for temp_str in os.listdir(DATA_PARENT):
    temp_dir = os.path.join(DATA_PARENT, temp_str)
    if not os.path.isdir(temp_dir) or not temp_str.isdigit():
        continue
    temperature = int(temp_str)
    for path in glob.glob(os.path.join(temp_dir, '*.csv')):
        m = iter_re.match(os.path.basename(path))
        if not m:
            continue
        iteration = m.group(1)
        try:
            df = pd.read_csv(path, skiprows=88)
        except EmptyDataError:
            continue
        if df.empty:
            continue
        df['Temperature'] = temperature
        df['Iteration']   = iteration
        frames.append(df)
if not frames:
    raise FileNotFoundError(f"No valid CSVs under {DATA_PARENT}")
df_all_energy = pd.concat(frames, ignore_index=True)

# --- Energy dashboard preprocessing (from Energy.py) ---
# Rename columns and add derived columns as in Energy.py
try:
    df_all_energy['Time'] = pd.to_datetime(df_all_energy['Scan Sweep Time (Sec)'])
    df_all_energy['T_aire_C'] = df_all_energy['116 (Vdc)- T_aire'] * 0.05
    df_all_energy['T_aire_K'] = df_all_energy['T_aire_C'] + 273.15
    df_all_energy['V_aire']   = df_all_energy['111 (Vdc)- V_aire'] * 0.025
    PIPE_DIAMETER_M = 0.1082
    PIPE_AREA       = np.pi * (PIPE_DIAMETER_M/2)**2
    ATM_PRESSURE    = 101325
    PRESSURE_CONV   = 1.0
    df_all_energy['rho_air']        = df_all_energy['T_aire_K'].apply(
        lambda T: PropsSI('D','T',T,'P',ATM_PRESSURE,'Air')
    )
    df_all_energy['mass_flow_rate'] = df_all_energy['rho_air'] * PIPE_AREA * df_all_energy['V_aire']
    T1_col     = next(c for c in df_all_energy.columns if c.strip().endswith('- T1'))
    T30_col    = next(c for c in df_all_energy.columns if c.strip().endswith('- T30'))
    p_diff_col = next(c for c in df_all_energy.columns if 'P_diff' in c)
    # Apply correct conversion for delta P (Pa) from sensor 121 (Adc)- P_diff
    # ΔP[Pa] = 100 * (P_diff * 2500 - 10)
    df_all_energy['dP_Pa'] = 100 * (df_all_energy[p_diff_col] * 2500 - 10)
    df_all_energy['T1_C']   = df_all_energy[T1_col]
    df_all_energy['T30_C']  = df_all_energy[T30_col]
    df_all_energy['deltaT'] = df_all_energy['T30_C'] - df_all_energy['T1_C']
    df_all_energy['mid_T_K'] = (df_all_energy['T1_C'] + df_all_energy['T30_C']) / 2 + 273.15
    df_all_energy['cp_air']  = df_all_energy['mid_T_K'].apply(
        lambda T: PropsSI('Cpmass','T',T,'P',ATM_PRESSURE,'Air')
    )
    df_all_energy['energy_rate'] = (
        df_all_energy['mass_flow_rate']
      * df_all_energy['cp_air']
      * (df_all_energy['T1_C'] - df_all_energy['T30_C'])
    )
    df_all_energy['fan_power'] = PIPE_AREA * df_all_energy['V_aire'] * df_all_energy['dP_Pa']
    df_all_energy.sort_values(['Temperature','Iteration','Time'], inplace=True)
    df_all_energy['dt'] = df_all_energy.groupby(['Temperature','Iteration'])['Time']\
                         .diff().dt.total_seconds().fillna(0)
    df_all_energy['dE_air'] = df_all_energy['energy_rate'] * df_all_energy['dt']
    df_all_energy['dW_fan'] = df_all_energy['fan_power']  * df_all_energy['dt']
    df_all_energy['E_air_cum'] = df_all_energy.groupby(['Temperature','Iteration'])['dE_air'].cumsum()
    df_all_energy['W_fan_cum'] = df_all_energy.groupby(['Temperature','Iteration'])['dW_fan'].cumsum()
    df_all_energy['E_st_cum']  = df_all_energy['E_air_cum'] - df_all_energy['W_fan_cum']
except Exception as e:
    print(f"Energy dashboard preprocessing error: {e}")

def main_layout():
    return html.Div([
        html.Div([
            html.Label("Select Dashboard:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Dropdown(
                id="dashboard-selector",
                options=DASHBOARD_OPTIONS,
                value="temperature",
                clearable=False,
                style={"width": "300px", "display": "inline-block"}
            )
        ], style={"marginBottom": "30px", "marginTop": "20px", "textAlign": "center"}),
        html.Div(id="dashboard-content"),
    ])

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.layout = main_layout

@app.callback(
    Output("dashboard-content", "children"),
    Input("dashboard-selector", "value")
)
def display_dashboard(selected):
    if selected == "energy":
        return energy_dashboard_layout(df_all_energy)
    return temperature_dashboard_layout()

# -------------------
# FULL CALLBACKS AND HELPERS FOR TEMPERATURE DASHBOARD
# -------------------
import os
import re
import pandas as pd
import numpy as np
import scipy.interpolate
import plotly.graph_objects as go
from dash import ctx

def load_data(temp: str, iter_file: str) -> pd.DataFrame:
    """
    Reads the CSV in data/<temp>/<iter_file>, skipping the first 88 rows,
    renames columns like '101 (°C)- T1' → 'T1', '111 (V)- T111', '116 (V)- T116', etc.
    Filters out T_ext1 > 100 as before.
    """
    path = os.path.join("data", str(temp), iter_file)
    df = pd.read_csv(path, encoding="utf-8", header=88)
    df.columns = df.columns.str.strip()
    rename_map = {}
    for col in df.columns:
        raw = col.strip()
        # If it begins with 111 or 116:
        m_num = re.match(r"^(\d+)", raw)
        if m_num:
            num = m_num.group(1)
            if num == "111":
                rename_map[col] = "T111"
            elif num == "116":
                rename_map[col] = "T116"
        # If “- T_extX”
        m_ext = re.search(r"-\s*(T_ext\d+)$", raw)
        if m_ext:
            rename_map[col] = m_ext.group(1)
        # If “- T<digits>”
        m_gen = re.search(r"-\s*(T\d+)$", raw)
        if m_gen:
            rename_map[col] = m_gen.group(1)
        # Already “T1”, “T2”, etc.
        if re.fullmatch(r"T(?:ext\d+|\d+)$", raw):
            rename_map[col] = raw
    df = df.rename(columns=rename_map)
    if "Scan Sweep Time (Sec)" in df.columns:
        df["Scan Sweep Time (Sec)"] = pd.to_datetime(
            df["Scan Sweep Time (Sec)"], errors="coerce"
        )
    if "Scan Number" in df.columns:
        df["Scan Number"] = pd.to_numeric(df["Scan Number"], errors="coerce")
    if "T_ext1" in df.columns:
        df["T_ext1"] = pd.to_numeric(df["T_ext1"], errors="coerce")
    if 'P_diff' in df.columns:
        df['P_diff'] = 100 * (df['P_diff'] * 2500 - 10)
    elif 'P_diff (Pa)' in df.columns:
        df['P_diff (Pa)'] = 100 * (df['P_diff (Pa)'] * 2500 - 10)
    return df

def categorize_sensors(df: pd.DataFrame) -> dict:
    groups = {
        "Thermocouples": [],
        "Wall Temps": [],
        "Air Temp": [],
        "Air Velocity": [],
        "External Temps": [],
        "Flow Sensors": [],
        "Pressure Sensors": [],
        "Other": []
    }
    for col in df.columns:
        name = col.strip()
        # Air velocity sensor
        if name == "T111" or "V_aire" in name:
            groups["Air Velocity"].append(name)
        elif name == "T116" or ("T_aire" in name and "Vdc" in name):
            groups["Air Temp"].append(name)
        elif re.fullmatch(r"T\d{1,2}$", name):
            groups["Thermocouples"].append(name)
        elif "pared" in name.lower():
            groups["Wall Temps"].append(name)
        elif "aire" in name.lower() and name != "T116":
            groups["Air Temp"].append(name)
        elif re.fullmatch(r"T_ext\d+", name) or "T_ext" in name:
            groups["External Temps"].append(name)
        elif "flux" in name.lower():
            groups["Flow Sensors"].append(name)
        elif "p_diff" in name.lower() or "P_diff" in name:
            groups["Pressure Sensors"].append(name)
        else:
            groups["Other"].append(name)
    # Remove empty groups
    return {k: v for k, v in groups.items() if v}

def make_cylinder_surface(
    df: pd.DataFrame,
    temp_cols: list,
    time_idx: int,
    selected_radii: list
) -> go.Figure:
    # Direct copy from Temperature.py with green color scale, colorbar, and hover
    final = df[temp_cols].iloc[time_idx].astype(float).to_numpy()
    n = len(temp_cols)
    if n == 0:
        return go.Figure().update_layout(title="No Thermocouples Found")
    layer_z = np.array([267.7, 201.7, 135.7, 69.7])
    temp_cols_sorted = sorted(temp_cols, key=lambda x: int(x[1:]))
    idx_map = {name: i for i, name in enumerate(temp_cols_sorted)}
    trees = [
        ["T10", "T11", "T12", "T13"],
        ["T6",  "T7",  "T8",  "T9"],
        ["T2",  "T3",  "T4",  "T5"],
        ["T26", "T27", "T28", "T29"],
        ["T22", "T23", "T24", "T25"],
        ["T14", "T15", "T16", "T17"],
        ["T18", "T19", "T20", "T21"]
    ]
    r1 = 892.0 / 2
    r2 = 699.2 / 2
    r3 = 506.6 / 2
    tree_xy = [
        (0.0, r1),
        (0.0, r2),
        (0.0, 157.0)
    ]
    angle_left  = np.deg2rad(210)
    tree_xy.append((r2 * np.cos(angle_left), r2 * np.sin(angle_left)))
    tree_xy.append((r3 * np.cos(angle_left), r3 * np.sin(angle_left)))
    angle_right = np.deg2rad(330)
    tree_xy.append((r3 * np.cos(angle_right), r3 * np.sin(angle_right)))
    tree_xy.append((r1 * np.cos(angle_right), r1 * np.sin(angle_right)))
    tree_angles_for_top = []
    for (x0, y0) in tree_xy:
        θ = np.arctan2(y0, x0) % (2*np.pi)
        tree_angles_for_top.append(θ)
    θ_grid = np.linspace(0, 2*np.pi, 100)
    T_grid = np.zeros((len(θ_grid), len(layer_z)))
    for iz, _ in enumerate(layer_z):
        node_theta = []
        node_temps = []
        for tree_idx, tree in enumerate(trees):
            if tree[iz] in idx_map:
                node_theta.append(tree_angles_for_top[tree_idx])
                node_temps.append(final[idx_map[tree[iz]]])
        θ_arr = np.array(node_theta)
        T_arr = np.array(node_temps)
        sort_idx = np.argsort(θ_arr)
        θ_sorted = θ_arr[sort_idx]
        T_sorted = T_arr[sort_idx]
        θ_ext = np.concatenate([
            θ_sorted[-1:] - 2*np.pi,
            θ_sorted,
            θ_sorted[:1] + 2*np.pi
        ])
        T_ext = np.concatenate([
            T_sorted[-1:], T_sorted, T_sorted[:1]
        ])
        f_interp = scipy.interpolate.interp1d(θ_ext, T_ext, kind='linear')
        T_grid[:, iz] = f_interp(θ_grid)
    T1_val  = float(df["T1"].iloc[time_idx]) if "T1" in df.columns else None
    T30_val = float(df["T30"].iloc[time_idx]) if "T30" in df.columns else None
    fig = go.Figure()
    R_outer = 1134 / 2
    for ang in np.linspace(0, 2*np.pi, 24, endpoint=False):
        x_line = R_outer * np.cos(ang) * np.ones(2)
        y_line = R_outer * np.sin(ang) * np.ones(2)
        z_line = np.array([0.0, 334.0])
        fig.add_trace(
            go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines', line=dict(color='gray', width=1),
                showlegend=False
            )
        )
    θ_circle = np.linspace(0, 2*np.pi, 80)
    fig.add_trace(
        go.Scatter3d(
            x=R_outer * np.cos(θ_circle),
            y=R_outer * np.sin(θ_circle),
            z=np.zeros_like(θ_circle),
            mode='lines', line=dict(color='gray', width=1),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=R_outer * np.cos(θ_circle),
            y=R_outer * np.sin(θ_circle),
            z=np.full_like(θ_circle, 334.0),
            mode='lines', line=dict(color='gray', width=1),
            showlegend=False
        )
    )
    for radius, dash_style, zface in [
        (r1, "dash", 0.0), (r2, "dash", 0.0), (157.0, "dash", 0.0), (r3, "solid", 0.0),
        (r1, "dash", 334.0), (r2, "dash", 334.0), (157.0, "dash", 334.0), (r3, "solid", 334.0)
    ]:
        style = dict(color="black", width=2)
        if dash_style == "dash":
            style["dash"] = "dash"
        fig.add_trace(
            go.Scatter3d(
                x=radius * np.cos(θ_circle),
                y=radius * np.sin(θ_circle),
                z=np.full_like(θ_circle, zface),
                mode="lines", line=style,
                showlegend=False
            )
        )
    r4 = 58.0 / 2
    fig.add_trace(
        go.Scatter3d(
            x=r4 * np.cos(θ_circle),
            y=r4 * np.sin(θ_circle),
            z=np.full_like(θ_circle, -30.0),
            mode="lines", line=dict(color="black", width=2),
            showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=r4 * np.cos(θ_circle),
            y=r4 * np.sin(θ_circle),
            z=np.full_like(θ_circle, 364.0),
            mode="lines", line=dict(color="black", width=2),
            showlegend=False
        )
    )
    # 9d) Thermocouple rings (only selected radii visible)
    unique_radii = sorted({np.hypot(x0, y0) for (x0, y0) in tree_xy})
    clim_min = float(np.min(T_grid))
    clim_max = float(np.max(T_grid))
    marker_size = 7  # smaller balls
    ring_line_width = 2  # thinner rings
    for iz, zval in enumerate(layer_z):
        for r_idx, r_ring in enumerate(unique_radii):
            if r_ring not in selected_radii:
                continue
            θ_ring = np.linspace(0, 2*np.pi, 200)
            x_ring = r_ring * np.cos(θ_ring)
            y_ring = r_ring * np.sin(θ_ring)
            z_ring = np.full_like(θ_ring, zval)
            # If this is the innermost ring (157 mm), use the corresponding thermocouple value for the whole ring
            if np.isclose(r_ring, 157.0):
                # Map layer to T2, T3, T4, T5 (top to bottom)
                tc_names = ["T2", "T3", "T4", "T5"]
                tc_name = tc_names[iz] if iz < len(tc_names) and tc_names[iz] in df.columns else None
                if tc_name:
                    tc_val = float(df[tc_name].iloc[time_idx])
                    T_ring = np.full_like(θ_ring, tc_val)
                else:
                    T_ring = np.full_like(θ_ring, np.nan)
            else:
                T_ring = T_grid[:, iz] if T_grid.shape[0] == len(θ_ring) else np.interp(np.linspace(0, 1, 200), np.linspace(0, 1, T_grid.shape[0]), T_grid[:, iz])
            fig.add_trace(
                go.Scatter3d(
                    x=x_ring, y=y_ring, z=z_ring,
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=T_ring,
                        colorscale="Viridis",
                        cmin=clim_min,
                        cmax=clim_max,
                        symbol="circle",
                        colorbar=dict(title="Temp (°C)", thickness=20) if iz == 0 and r_idx == 0 else None,
                        showscale=True if iz == 0 and r_idx == 0 else False
                    ),
                    showlegend=False,
                    hovertemplate="Temp: %{marker.color:.1f}°C<br>x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}"
                )
            )
    # Add solid red vertical bars, red rings (parallel to ground), and bold black labels at thermocouple locations
    tc_label_font = dict(size=24, color="black", family="Arial", weight="bold")
    ring_radius = 9  # smaller red rings
    n_ring_pts = 40
    for tree_idx, tree in enumerate(trees):
        x0, y0 = tree_xy[tree_idx]
        zvals = list(layer_z)
        # Draw solid red vertical bar connecting all thermocouple z positions for this tree
        fig.add_trace(
            go.Scatter3d(
                x=[x0]*len(zvals),
                y=[y0]*len(zvals),
                z=zvals,
                mode="lines",
                line=dict(color="red", width=ring_line_width),
                showlegend=False
            )
        )
        # Add red ring (parallel to ground) and bold black label at each thermocouple position
        for layer_i, zval in enumerate(layer_z):
            tc_name = tree[layer_i]
            tc_temp = float(df[tc_name].iloc[time_idx]) if tc_name in df.columns else None
            # Red ring in xy-plane at (x0, y0, zval)
            theta = np.linspace(0, 2*np.pi, n_ring_pts)
            x_ring = x0 + ring_radius * np.cos(theta)
            y_ring = y0 + ring_radius * np.sin(theta)
            z_ring = np.full_like(theta, zval)
            fig.add_trace(
                go.Scatter3d(
                    x=x_ring, y=y_ring, z=z_ring,
                    mode="lines",
                    line=dict(color="red", width=ring_line_width),
                    showlegend=False,
                    hovertemplate=f"{tc_name}: {tc_temp:.1f}°C<br>x: {x0:.1f}<br>y: {y0:.1f}<br>z: {zval:.1f}"
                )
            )
            # Bold black label at each thermocouple position
            fig.add_trace(
                go.Scatter3d(
                    x=[x0], y=[y0], z=[zval],
                    mode="text",
                    text=[tc_name],
                    textfont=tc_label_font,
                    showlegend=False
                )
            )
    # Add T1 perfectly centered on the innermost black circle at z=364, and T30 perfectly centered on the innermost black circle at z=-30
    r_black = 29.0  # innermost black circle radius (mm)
    t1_z = 364.0
    t30_z = -30.0
    # Place at the center of the circle: x=0, y=0
    t1_x, t1_y = 0.0, 0.0
    t30_x, t30_y = 0.0, 0.0
    t1_temp = float(df["T1"].iloc[time_idx]) if "T1" in df.columns else None
    t30_temp = float(df["T30"].iloc[time_idx]) if "T30" in df.columns else None
    # T1 (centered at x=0, y=0, z=364)
    fig.add_trace(
        go.Scatter3d(
            x=[t1_x], y=[t1_y], z=[t1_z],
            mode="markers+text",
            marker=dict(
                size=marker_size,
                color=[t1_temp],
                colorscale="Viridis",
                cmin=clim_min,
                cmax=clim_max,
                symbol="circle"
            ),
            text=["T1"],
            textposition="top center",
            textfont=dict(size=22, color="black", family="Arial", weight="bold"),
            showlegend=False,
            hovertemplate=f"T1: {t1_temp:.1f}°C<br>x: {t1_x:.1f}<br>y: {t1_y:.1f}<br>z: {t1_z:.1f}"
        )
    )
    # T30 (centered at x=0, y=0, z=-30)
    fig.add_trace(
        go.Scatter3d(
            x=[t30_x], y=[t30_y], z=[t30_z],
            mode="markers+text",
            marker=dict(
                size=marker_size,
                color=[t30_temp],
                colorscale="Viridis",
                cmin=clim_min,
                cmax=clim_max,
                symbol="circle"
            ),
            text=["T30"],
            textposition="top center",
            textfont=dict(size=22, color="black", family="Arial", weight="bold"),
            showlegend=False,
            hovertemplate=f"T30: {t30_temp:.1f}°C<br>x: {t30_x:.1f}<br>y: {t30_y:.1f}<br>z: {t30_z:.1f}"
        )
    )
    camera = dict(eye=dict(x=1.2, y=1.2, z=0.9))
    H_total = 334.0
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=H_total / (2 * R_outer))
        ),
        scene_camera=camera,
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Thermal Cylinder @ Row {time_idx}",
        height=800
    )
    return fig

@app.callback(
    Output("iter-dropdown", "options"),
    Output("iter-dropdown", "value"),
    Output("cyl-iter-dropdown", "options"),
    Output("cyl-iter-dropdown", "value"),
    Input("temp-dropdown", "value")
)
def update_iterations(temp):
    temp_dir = os.path.join("data", str(temp))
    if not os.path.isdir(temp_dir):
        return [], [], [], None
    files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]
    options = [{"label": f, "value": f} for f in files]
    value = files[:1] if files else []
    return options, value, options, (files[0] if files else None)

@app.callback(
    Output("group-dropdown", "options"),
    Output("group-dropdown", "value"),
    Input("temp-dropdown", "value"),
    Input("iter-dropdown", "value")
)
def update_groups(temp, iter_files):
    if not iter_files:
        return [], None
    df = load_data(temp, iter_files[0])
    groups = categorize_sensors(df)
    group_options = [{"label": k, "value": k} for k in groups if groups[k]]
    group_value = group_options[0]["value"] if group_options else None
    return group_options, group_value

@app.callback(
    Output("sensor-dropdown", "options"),
    Output("sensor-dropdown", "value"),
    Input("group-dropdown", "value"),
    Input("select-all-btn", "n_clicks"),
    Input("clear-all-btn", "n_clicks"),
    Input("temp-dropdown", "value"),
    Input("iter-dropdown", "value")
)
def update_sensors(group, sel_all, clr_all, temp, iter_files):
    if not group or not iter_files:
        return [], []
    df = load_data(temp, iter_files[0])
    groups = categorize_sensors(df)
    sensors = groups.get(group, [])
    triggered = ctx.triggered_id
    auto_all = ["Wall Temps", "External Temps", "Flow Sensors", "Pressure Sensors"]
    if group in auto_all:
        return [{"label": s, "value": s} for s in sensors], sensors
    if triggered == "select-all-btn":
        return [{"label": s, "value": s} for s in sensors], sensors
    elif triggered == "clear-all-btn":
        return [{"label": s, "value": s} for s in sensors], []
    return [{"label": s, "value": s} for s in sensors], sensors[:1] if sensors else []

@app.callback(
    Output("sensor-graph", "figure"),
    Input("sensor-dropdown", "value"),
    Input("group-dropdown", "value"),
    Input("temp-dropdown", "value"),
    Input("iter-dropdown", "value")
)
def update_graph(sensors, group, temp, iter_files):
    from plotly.subplots import make_subplots
    import plotly.graph_objs as go
    if not sensors or not iter_files:
        return go.Figure()
    # Wall Temps: always show both pared sensors
    if group == "Wall Temps":
        pared_sensors = [s for s in sensors if "pared" in s.lower()]
        sensors = pared_sensors if pared_sensors else sensors
    # Axis labels with units
    y_label = ""
    x_label = "Time (s)"
    if group == "Thermocouples":
        y_label = "Temperature (°C)"
    elif group == "Wall Temps":
        y_label = "Wall Temp (°C)"
    elif group == "Air Temp":
        y_label = "Air Temp (°C)"
    elif group == "Air Velocity":
        y_label = "Velocity (m/s)"
    elif group == "External Temps":
        y_label = "External Temp (°C)"
    elif group == "Flow Sensors":
        y_label = "Flow (L/min)"
    elif group == "Pressure Sensors":
        y_label = "Pressure (Pa)"
    else:
        y_label = "Output (V)"
    # Air Velocity and Air Temp: special handling
    if group == "Air Velocity":
        fig = go.Figure()
        for iter_file in iter_files:
            df = load_data(temp, iter_file)
            for s in sensors:
                if "111" in s or "V_aire" in s:
                    v = df[s] if s in df.columns else None
                    if v is not None:
                        velocity = v * 0.025
                        fig.add_trace(go.Scatter(
                            x=velocity,
                            y=v,
                            mode="markers",
                            marker=dict(size=8),
                            name=f"{s} ({iter_file})"
                        ))
        fig.update_layout(title="Air Velocity: Voltage vs Velocity", xaxis_title="Velocity (m/s)", yaxis_title="Voltage (V)", height=600)
        return fig
    if group == "Air Temp":
        fig = go.Figure()
        for iter_file in iter_files:
            df = load_data(temp, iter_file)
            for s in sensors:
                if "116" in s or "T_aire" in s:
                    v = df[s] if s in df.columns else None
                    if v is not None:
                        tempC = v * 0.05
                        fig.add_trace(go.Scatter(
                            x=tempC,
                            y=v,
                            mode="markers",
                            marker=dict(size=8),
                            name=f"{s} ({iter_file})"
                        ))
        fig.update_layout(title="Air Temperature: Voltage vs Temperature", xaxis_title="Temperature (°C)", yaxis_title="Voltage (V)", height=600)
        return fig
    # Default: facet by iteration, one graph per iteration, autoscaled
    fig = make_subplots(rows=1, cols=len(iter_files), subplot_titles=iter_files)
    for i, iter_file in enumerate(iter_files):
        df = load_data(temp, iter_file)
        for s in sensors:
            if s in df.columns:
                if group == "Thermocouples":
                    fig.add_trace(go.Scatter(
                        x=df["Scan Sweep Time (Sec)"] if "Scan Sweep Time (Sec)" in df.columns else df.index,
                        y=df[s],
                        mode="lines",
                        name=s
                    ), row=1, col=i+1)
                else:
                    fig.add_trace(go.Scatter(
                        x=df["Scan Sweep Time (Sec)"] if "Scan Sweep Time (Sec)" in df.columns else df.index,
                        y=df[s],
                        mode="markers",
                        marker=dict(size=8),
                        name=s
                    ), row=1, col=i+1)
        fig.update_xaxes(autorange=True, row=1, col=i+1, title_text=x_label)
        fig.update_yaxes(autorange=True, row=1, col=i+1, title_text=y_label)
    fig.update_layout(title=f"Sensor Readings – {temp}°C | {', '.join(iter_files)}", showlegend=True, height=600)
    return fig

@app.callback(
    Output("time-slider", "min"),
    Output("time-slider", "max"),
    Output("time-slider", "marks"),
    Output("time-slider", "value"),
    Output("time-index-input", "min"),
    Output("time-index-input", "max"),
    Output("time-index-input", "value"),
    Input("temp-dropdown", "value"),
    Input("cyl-iter-dropdown", "value"),
    Input("time-slider", "value"),
    Input("time-index-input", "value")
)
def update_time_slider_and_sync(temp, cyl_iter, slider_val, input_val):
    if not cyl_iter:
        return 0, 0, {}, 0, 0, 0, 0
    df = load_data(temp, cyl_iter)
    n = len(df)
    marks = {i: str(i) for i in range(0, n, max(1, n // 10))}
    value = slider_val if slider_val is not None else 0
    value = min(value, n-1) if n > 0 else 0
    return 0, n-1, marks, value, 0, n-1, value

@app.callback(
    Output("cylinder-3d", "figure"),
    Input("temp-dropdown", "value"),
    Input("cyl-iter-dropdown", "value"),
    Input("time-slider", "value"),
    Input("ring-select", "value")
)
def update_cylinder(temp, cyl_iter, time_idx, ring_select):
    if not cyl_iter:
        return go.Figure()
    df = load_data(temp, cyl_iter)
    temp_cols = [col for col in df.columns if re.fullmatch(r"T\d{1,2}", col)]
    return make_cylinder_surface(df, temp_cols, time_idx or 0, ring_select or [])

# -------------------
# FULL CALLBACKS AND HELPERS FOR ENERGY DASHBOARD
# -------------------
# (All callback and helper function code from Energy.py, with IDs prefixed by 'energy-')

# Helper to break axis-linking in facet plots

def break_linking(fig):
    fig.for_each_xaxis(lambda ax: ax.update(matches=None))
    fig.for_each_yaxis(lambda ax: ax.update(matches=None))

# Populate iteration-dropdown on temperature change
@app.callback(
    Output('energy-iteration-dropdown','options'),
    Output('energy-iteration-dropdown','value'),
    Input('energy-temp-dropdown','value')
)
def set_iterations_energy(temp):
    its = sorted(df_all_energy[df_all_energy['Temperature']==temp]['Iteration'].unique(), key=int)
    opts = [{'label': f"Iter {i}", 'value': i} for i in its]
    return opts, [its[0]] if its else []

# Helper to adjust discharge energy so it starts from zero at the transition point and is downward-pointing

def adjust_discharge_energy(df, energy_col='E_air_cum', phase_col='phase'):
    df = df.copy()
    charge_mask = df[phase_col] == 'Charge'
    discharge_mask = df[phase_col] == 'Discharge'
    if not discharge_mask.any() or not charge_mask.any():
        return df[energy_col]
    last_charge_idx = df[charge_mask].index[-1]
    max_charge_energy = df.loc[last_charge_idx, energy_col]
    adjusted = df[energy_col].copy()
    # Downward-pointing: subtract from max_charge_energy
    adjusted[discharge_mask] = max_charge_energy - adjusted[discharge_mask]
    # Shift discharge curve up so its first value matches the last charge value
    if discharge_mask.any():
        first_discharge_idx = df[discharge_mask].index[0]
        shift_val = max_charge_energy - adjusted[first_discharge_idx]
        adjusted[discharge_mask] += shift_val
    return adjusted

# --- Helper to robustly connect charge and discharge for each iteration ---
def connect_charge_discharge(df, energy_col='E_air_cum', phase_col='phase', time_col='Time', iteration_col='Iteration'):
    """
    For each iteration, ensure the last charge point and first discharge point are connected
    by duplicating the transition point at the start of discharge if needed.
    Returns a new DataFrame with connected segments for all iterations.
    """
    dfs = []
    for it in df[iteration_col].unique():
        dfi = df[df[iteration_col] == it].copy()
        charge_mask = dfi[phase_col] == 'Charge'
        discharge_mask = dfi[phase_col] == 'Discharge'
        if not charge_mask.any() or not discharge_mask.any():
            dfs.append(dfi)
            continue
        # Find transition index
        last_charge_idx = dfi[charge_mask].index[-1]
        first_discharge_idx = dfi[discharge_mask].index[0]
        # If the transition point is not the same, duplicate the last charge point at the start of discharge
        if dfi.loc[last_charge_idx, time_col] != dfi.loc[first_discharge_idx, time_col]:
            # Insert the last charge point at the start of discharge
            row = dfi.loc[[last_charge_idx]].copy()
            row[phase_col] = 'Discharge'
            # Insert before first_discharge_idx
            dfi = pd.concat([
                dfi.loc[:last_charge_idx],
                row,
                dfi.loc[first_discharge_idx:]
            ]).reset_index(drop=True)
        dfs.append(dfi)
    return pd.concat(dfs, ignore_index=True)

# --- Update all figures, using robust connection for charge/discharge ---
@app.callback(
    Output('energy-mass-flow-graph','figure'),
    Output('energy-deltaT-graph','figure'),
    Output('energy-energy-graph','figure'),
    Output('energy-stored-energy-graph','figure'),
    Output('energy-fan-work-graph','figure'),
    Output('energy-efficiency-graph','figure'),
    Input('energy-temp-dropdown','value'),
    Input('energy-iteration-dropdown','value'),
)
def update_all_energy(temp, iterations):
    if not iterations:
        return ({},) * 6
    d = df_all_energy[
        (df_all_energy['Temperature']==temp) &
        (df_all_energy['Iteration'].isin(iterations))
    ]
    # 1) Mass Flow
    fig1 = px.line(
        d, x='Time', y='mass_flow_rate',
        facet_col='Iteration', facet_col_wrap=2,
        title=f"Mass Flow Rate @ {temp}°C",
        labels={'mass_flow_rate':'kg/s'}
    )
    break_linking(fig1)
    # 2) ΔT
    fig2 = px.line(
        d, x='Time', y='deltaT',
        facet_col='Iteration', facet_col_wrap=2,
        title=f"Outlet–Inlet ΔT @ {temp}°C",
        labels={'deltaT':'ΔT (°C)'}
    )
    break_linking(fig2)
    # 3) Cumulative Air Energy with robust connection
    switch_times = {}
    for it in iterations:
        dd = d[d['Iteration']==it]
        jump = dd['energy_rate'].diff().abs()
        idx  = jump.idxmax()
        switch_times[it] = dd.loc[idx, 'Time']
    d2 = d.copy()
    d2['phase'] = [
        'Charge' if t <= switch_times[it] else 'Discharge'
        for it, t in zip(d2['Iteration'], d2['Time'])
    ]
    d2['E_air_cum_adj'] = adjust_discharge_energy(d2)
    # Shift discharge so it connects to the end of charge for each iteration
    d2['E_air_cum_shifted'] = shift_discharge_energy_to_connect(d2, energy_col='E_air_cum_adj', phase_col='phase', iteration_col='Iteration')
    # Connect charge/discharge for each iteration (for plotting)
    d2c = connect_charge_discharge(d2, energy_col='E_air_cum_shifted', phase_col='phase', time_col='Time', iteration_col='Iteration')
    fig3 = px.line(
        d2c, x='Time', y='E_air_cum_shifted', color='phase',
        facet_col='Iteration', facet_col_wrap=2, markers=True,
        title=f"Cumulative Air Energy @ {temp}°C (Discharge Shifted, Connected)",
        labels={'E_air_cum_shifted':'Energy (J)','phase':'Phase'}
    )
    break_linking(fig3)
    # 4) Stored Energy: combine charge and discharge adjusted curves for visual match, robustly connected
    d2c['E_stored_combined'] = d2c['E_air_cum_shifted']
    fig4 = px.line(
        d2c, x='Time', y='E_stored_combined',
        facet_col='Iteration', facet_col_wrap=2, markers=True,
        title=f"Stored Energy @ {temp}°C (Combined Charge+Discharge, Connected)",
        labels={'E_stored_combined':'Stored Energy (J)'}
    )
    break_linking(fig4)
    # 5) Fan Work
    fig5 = px.line(
        d, x='Time', y='W_fan_cum',
        facet_col='Iteration', facet_col_wrap=2, markers=True,
        title=f"Fan Work @ {temp}°C",
        labels={'W_fan_cum':'Fan Work (J)'}
    )
    break_linking(fig5)
    # 6) Efficiencies with zero-division guards
    rows = []
    for it in iterations:
        dd_energy = d2c[d2c['Iteration']==it]
        charge_mask = dd_energy['phase'] == 'Charge'
        discharge_mask = dd_energy['phase'] == 'Discharge'
        # Use raw unadjusted cumulative energy and fan work for charge phase
        E_in = dd_energy.loc[charge_mask, 'dE_air'].sum()
        W_fan_ch = dd_energy.loc[charge_mask, 'dW_fan'].sum()
        # True stored energy = energy in minus fan losses (during charge)
        E_st = E_in - W_fan_ch
        # E_ch is just the energy input
        E_ch = E_in
        # Discharge energy: sum of dE_air after the transition point
        if discharge_mask.any():
            E_discharge = dd_energy.loc[discharge_mask, 'dE_air'].sum()
        else:
            E_discharge = 0.0
        # Charging efficiency: stored / input
        eta_ch = 0.0 if E_ch == 0 else E_st / E_ch
        # Discharging efficiency: discharge / stored
        eta_dch = 0.0 if E_st == 0 else abs(E_discharge) / E_st
        # Round-trip efficiency: discharge / input
        eta_rt = 0.0 if E_ch == 0 else abs(E_discharge) / E_ch
        rows += [
            {'Iteration':it,'Metric':'Round-trip',  'η':eta_rt},
            {'Iteration':it,'Metric':'Charging',    'η':eta_ch},
            {'Iteration':it,'Metric':'Discharging', 'η':eta_dch},
        ]
    # Ensure all metrics are present for all iterations, even if 0
    metrics = ['Round-trip', 'Charging', 'Discharging']
    for it in iterations:
        for m in metrics:
            if not any((row['Iteration']==it and row['Metric']==m) for row in rows):
                rows.append({'Iteration':it, 'Metric':m, 'η':0.0})
    df_eff = pd.DataFrame(rows)
    fig6 = px.bar(
        df_eff, x='Metric', y='η',
        facet_col='Iteration', facet_col_wrap=2,
        title=f"Efficiencies @ {temp}°C",
        labels={'η':'Efficiency'}
    )
    break_linking(fig6)
    return fig1, fig2, fig3, fig4, fig5, fig6

# --- Helper to shift discharge energy so it connects to the end of charge ---
def shift_discharge_energy_to_connect(df, energy_col='E_air_cum', phase_col='phase', iteration_col='Iteration'):
    """
    For each iteration, shift the discharge energy curve so its first value matches the last value of the charge curve.
    Returns a new Series with the shifted energy values.
    """
    shifted = df[energy_col].copy()
    for it in df[iteration_col].unique():
        dfi = df[df[iteration_col] == it]
        charge_mask = dfi[phase_col] == 'Charge'
        discharge_mask = dfi[phase_col] == 'Discharge'
        if not charge_mask.any() or not discharge_mask.any():
            continue
        last_charge_val = dfi.loc[charge_mask, energy_col].iloc[-1]
        first_discharge_val = dfi.loc[discharge_mask, energy_col].iloc[0]
        shift = last_charge_val - first_discharge_val
        shifted.loc[dfi[discharge_mask].index] += shift
    return shifted

if __name__ == "__main__":
    app.run(debug=True)
