# app.py








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








# --------------------------------------------------
# 1) DATA LOADING / COLUMN RENAMING
# --------------------------------------------------








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
            num = int(m_num.group(1))
            if num == 111:
                rename_map[col] = "T111"
                continue
            elif num == 116:
                rename_map[col] = "T116"
                continue
        # If “- T_extX”
        m_ext = re.search(r"-\s*(T_ext\d+)$", raw)
        if m_ext:
            rename_map[col] = m_ext.group(1)
            continue
        # If “- T<digits>”
        m_gen = re.search(r"-\s*(T\d+)$", raw)
        if m_gen:
            rename_map[col] = m_gen.group(1)
            continue
        # Already “T1”, “T2”, etc.
        if re.fullmatch(r"T(?:ext\d+|\d+)$", raw):
            rename_map[col] = raw








    df = df.rename(columns=rename_map)








    # Convert “Scan Sweep Time (Sec)” to datetime if present
    if "Scan Sweep Time (Sec)" in df.columns:
        df["Scan Sweep Time (Sec)"] = pd.to_datetime(
            df["Scan Sweep Time (Sec)"], errors="coerce"
        )
    # Convert “Scan Number” to numeric if present
    if "Scan Number" in df.columns:
        df["Scan Number"] = pd.to_numeric(df["Scan Number"], errors="coerce")








    # Filter out any T_ext1 > 100
    if "T_ext1" in df.columns:
        df["T_ext1"] = pd.to_numeric(df["T_ext1"], errors="coerce")








    # Convert pressure sensor using the provided formula if present
    if 'P_diff' in df.columns:
        df['P_diff'] = 100 * (df['P_diff'] * 2500 - 10)
    elif 'P_diff (Pa)' in df.columns:
        # If already named with (Pa), still apply conversion to be sure
        df['P_diff (Pa)'] = 100 * (df['P_diff (Pa)'] * 2500 - 10)
    # ...existing code...
    return df








# --------------------------------------------------
# 2) SENSOR GROUPING (unchanged)
# --------------------------------------------------








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
        if col in ["Scan Sweep Time (Sec)", "Scan Number"]:
            continue
        name = col
        # Air velocity sensor
        if name == "T111":
            groups["Air Velocity"].append(name)
        # Air Temp sensor
        elif name == "T116":
            groups["Air Temp"].append(name)
        # Thermocouple if “T” + 1 or 2 digits
        elif re.fullmatch(r"T\d{1,2}$", name):
            groups["Thermocouples"].append(name)
        elif "pared" in name.lower():
            groups["Wall Temps"].append(name)
        elif "aire" in name.lower() and name != "T116":
            groups["Air Temp"].append(name)
        elif re.fullmatch(r"T_ext\d+", name):
            groups["External Temps"].append(name)
        elif "flux" in name.lower():
            groups["Flow Sensors"].append(name)
        elif "p_diff" in name.lower():
            groups["Pressure Sensors"].append(name)
        else:
            groups["Other"].append(name)
    return groups








# --------------------------------------------------
# 3) INTERPOLATE TEMPERATURES ON CYLINDER (Rings Only + T1/T30 spheres)
# --------------------------------------------------








def make_cylinder_surface(
    df: pd.DataFrame,
    temp_cols: list,
    time_idx: int,
    selected_radii: list
) -> go.Figure:
    """
    Build a 3D figure showing:
      - Interpolated temperatures on thermocouple rings at chosen time index.
      - Colored spheres at T1 (inlet) and T30 (outlet).
      - "Tree" stems and labels in red, doubled font size, bold, black.
      - Only rings in selected_radii are visible (others transparent).
    """








    # 1) Grab the row at “time_idx” of ring temperatures
    final = df[temp_cols].iloc[time_idx].astype(float).to_numpy()
    n = len(temp_cols)
    if n == 0:
        return go.Figure().update_layout(title="No Thermocouples Found")








    # 2) Node layers in Z (mm)
    layer_z = np.array([267.7, 201.7, 135.7, 69.7])  # (top → bottom)








    # 3) Sort temp_cols “T2…T29” → index map
    temp_cols_sorted = sorted(temp_cols, key=lambda x: int(x[1:]))
    idx_map = {name: i for i, name in enumerate(temp_cols_sorted)}








    # 4) Define the 7 “trees” and their 4 T-nodes per tree
    trees = [
        ["T10", "T11", "T12", "T13"],   # North @ 446 mm
        ["T6",  "T7",  "T8",  "T9"],   # North @ 349.6 mm
        ["T2",  "T3",  "T4",  "T5"],  # North @ 157 mm
        ["T26", "T27", "T28", "T29"],  # Left  @ 349.6 mm
        ["T22", "T23", "T24", "T25"],  # Left  @ 253.3 mm
        ["T14", "T15", "T16", "T17"],  # Right @ 253.3 mm
        ["T18", "T19", "T20", "T21"]   # Right @ 446 mm
    ]
    # 5) XY positions for each tree on the top face:
    r1 = 892.0 / 2       # 446 mm  (TC_N1)
    r2 = 699.2 / 2       # 349.6 mm (TC_N2)
    r3 = 506.6 / 2       # 253.3 mm (middle-left & right)
    tree_xy = [
        (0.0, r1),       # TC_N1
        (0.0, r2),       # TC_N2
        (0.0, 157.0)     # TC_N3 (explicit)
    ]
    # “Left-bottom” at 210°:
    angle_left  = np.deg2rad(210)
    tree_xy.append((r2 * np.cos(angle_left), r2 * np.sin(angle_left)))  # TC_L1
    tree_xy.append((r3 * np.cos(angle_left), r3 * np.sin(angle_left)))  # TC_L2
    # “Right-bottom” at 330°:
    angle_right = np.deg2rad(330)
    tree_xy.append((r3 * np.cos(angle_right), r3 * np.sin(angle_right)))  # TC_R1
    tree_xy.append((r1 * np.cos(angle_right), r1 * np.sin(angle_right)))  # TC_R2








    # 6) Compute each tree’s θ_i in [0, 2π)
    tree_angles_for_top = []
    for (x0, y0) in tree_xy:
        θ = np.arctan2(y0, x0) % (2*np.pi)
        tree_angles_for_top.append(θ)








    # 7) Build θ_grid and T_grid for the 4 layers
    θ_grid = np.linspace(0, 2*np.pi, 100)
    T_grid = np.zeros((len(θ_grid), len(layer_z)))  # (100×4)








    for iz, _ in enumerate(layer_z):
        node_theta = []
        node_temps = []
        for tree_idx, tree in enumerate(trees):
            T_name = tree[iz]
            idx = idx_map[T_name]
            temp_val = final[idx]
            θ = tree_angles_for_top[tree_idx]
            node_theta.append(θ)
            node_temps.append(temp_val)








        θ_arr = np.array(node_theta)
        T_arr = np.array(node_temps)
        sort_idx = np.argsort(θ_arr)
        θ_sorted = θ_arr[sort_idx]
        T_sorted = T_arr[sort_idx]








        # Extend angles for circular interpolation
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








    # 8) Retrieve T1 and T30 values (if present) for colored spheres
    T1_val  = float(df["T1"].iloc[time_idx]) if "T1" in df.columns else None
    T30_val = float(df["T30"].iloc[time_idx]) if "T30" in df.columns else None








    # 9) Build Plotly figure
    fig = go.Figure()








    # 9a) Outer frame (wireframe cylinder) for reference
    R_outer = 1134 / 2  # 567 mm
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
    # bottom ring
    fig.add_trace(
        go.Scatter3d(
            x=R_outer * np.cos(θ_circle),
            y=R_outer * np.sin(θ_circle),
            z=np.zeros_like(θ_circle),
            mode='lines', line=dict(color='gray', width=1),
            showlegend=False
        )
    )
    # top ring
    fig.add_trace(
        go.Scatter3d(
            x=R_outer * np.cos(θ_circle),
            y=R_outer * np.sin(θ_circle),
            z=np.full_like(θ_circle, 334.0),
            mode='lines', line=dict(color='gray', width=1),
            showlegend=False
        )
    )








    # 9b) Static concentric circles (top & bottom)
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








    # 9c) Inner tube/hole circles (r4 = 29 mm)
    r4 = 58.0 / 2  # 29 mm
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
    first_ring = True
    for iz, zval in enumerate(layer_z):
        for r_ring in unique_radii:
            X_ring = r_ring * np.cos(θ_grid)
            Y_ring = r_ring * np.sin(θ_grid)
            Z_ring = np.full_like(θ_grid, zval)
            T_ring = T_grid[:, iz]
            opacity = 1.0 if r_ring in selected_radii else 0.0
            # Only show temperature in hover, not as text label
            fig.add_trace(
                go.Scatter3d(
                    x=X_ring, y=Y_ring, z=Z_ring,
                    mode="markers",
                    marker=dict(
                        size=4,
                        symbol="circle",
                        color=T_ring,
                        colorscale="Viridis",
                        cmin=clim_min,
                        cmax=clim_max,
                        opacity=opacity,
                        showscale=first_ring,
                        colorbar=dict(title="Temp (°C)", thickness=20) if first_ring else None
                    ),
                    hovertemplate="T: %{marker.color:.1f}°C<br>x: %{x:.1f}<br>y: %{y:.1f}<br>z: %{z:.1f}<extra></extra>",
                    showlegend=False
                )
            )
            first_ring = False








    # 9e) T1 inlet & T30 outlet as colored spheres (if available)
    sphere_size = 8
    if T1_val is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[0], y=[0], z=[364.0],
                mode="markers",
                marker=dict(
                    size=sphere_size,
                    color=[T1_val],
                    colorscale="Inferno",
                    cmin=clim_min,
                    cmax=clim_max,
                    showscale=False
                ),
                showlegend=False,
                hovertemplate="T1: %{marker.color:.1f}°C"
            )
        )
    if T30_val is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[0], y=[0], z=[-30.0],
                mode="markers",
                marker=dict(
                    size=sphere_size,
                    color=[T30_val],
                    colorscale="Inferno",
                    cmin=clim_min,
                    cmax=clim_max,
                    showscale=False
                ),
                showlegend=False,
                hovertemplate="T30: %{marker.color:.1f}°C"
            )
        )








    # 9f) “Vertical stems” + red circles + labels for T2–T29
    # Double font size (28), bold, black
    r_tc = 21.5 / 2  # 10.75 mm for TC hole
    tc_color = "red"
    for tree_idx, tree in enumerate(trees):
        x0, y0 = tree_xy[tree_idx]
        # Vertical red stem
        fig.add_trace(
            go.Scatter3d(
                x=[x0, x0], y=[y0, y0],
                z=[69.7, 267.7],
                mode="lines",
                line=dict(color=tc_color, width=4),
                showlegend=False
            )
        )
        # Hollow red circles + labels at each layer (offset by +5 mm)
        for layer_i, zval in enumerate(layer_z):
            Tname = tree[layer_i]
            fig.add_trace(
                go.Scatter3d(
                    x=x0 + r_tc * np.cos(θ_circle),
                    y=y0 + r_tc * np.sin(θ_circle),
                    z=np.full_like(θ_circle, zval + 5.0),
                    mode="lines",
                    line=dict(color=tc_color, width=4),
                    showlegend=False
                )
            )
            # Label above circle: size=28, bold, black
            fig.add_trace(
                go.Scatter3d(
                    x=[x0], y=[y0], z=[zval + 8.0],
                    mode="text", text=[Tname],
                    textposition="bottom center",
                    textfont=dict(size=28, color="black", family="Arial Bold"),
                    showlegend=False
                )
            )








    # 9g) Add T1 & T30 labels (size=32, bold, black)
    if T1_val is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[0], y=[0], z=[364.0 + 8.0],
                mode="text", text=["T1"],
                textposition="bottom center",
                textfont=dict(size=32, color="black", family="Arial Bold"),
                showlegend=False
            )
        )
    if T30_val is not None:
        fig.add_trace(
            go.Scatter3d(
                x=[0], y=[0], z=[-30.0 - 8.0],
                mode="text", text=["T30"],
                textposition="top center",
                textfont=dict(size=32, color="black", family="Arial Bold"),
                showlegend=False
            )
        )








    # 10) Camera & layout
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
        title=f"Thermal Cylinder @ Row {time_idx}"
    )








    return fig








# --------------------------------------------------
# 4) DASH APP SETUP
# --------------------------------------------------








app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)








def serve_layout():
    """
    Page layout:
      • Dropdown: Experiment Temperature (°C)
      • Dropdown: Iterations (multi-select for sensor graphs)
      • Dropdown: Sensor Group → Sensor List
      • Graph: Sensor‐comparison (side-by-side autoscaled)
      • NEW Dropdown: Cylinder‐Iteration (single-select)
      • NEW Slider: Time‐Index for cylinder
      • NEW Checklist: Rings to display (toggle transparency)
      • Graph: Cylinder‐3D (rings only at chosen iteration & time)
    """
    return dbc.Container([
        # TITLE
        dbc.Row(dbc.Col(
            html.H1("TES Sensor Dashboard", className="text-center text-primary my-4")
        )),
        # CONTROLS + SENSOR GRAPH
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
        # NEW CONTROLS FOR CYLINDER
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
                        marks={},      # will be populated by callback
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
                    value=[446.0, 349.6, 253.3, 157.0],  # default all visible
                    labelStyle={"display": "block"}
                )
            ], width=4)
        ], className="mb-4"),
        # CYLINDER 3D GRAPH
        dbc.Row([
            dbc.Col(dcc.Graph(id="cylinder-3d"), width=12)
        ])
    ], fluid=True)








app.layout = serve_layout








# --------------------------------------------------
# 5) CALLBACKS
# --------------------------------------------------








@app.callback(
    Output("iter-dropdown", "options"),
    Output("iter-dropdown", "value"),
    Output("cyl-iter-dropdown", "options"),
    Output("cyl-iter-dropdown", "value"),
    Input("temp-dropdown", "value")
)
def update_iterations(temp):
    """
    Whenever experiment temp changes, repopulate both:
      • iter-dropdown (multi-select for sensor graphs)
      • cyl-iter-dropdown (single-select for 3D cylinder)
    """
    folder = os.path.join("data", str(temp))
    if not os.path.exists(folder):
        return [], [], [], None








    files = sorted([f for f in os.listdir(folder) if f.endswith(".csv")])








    options = [{"label": f.replace(".csv", ""), "value": f} for f in files]
    sensor_values = [options[0]["value"]] if options else []
    cyl_value = options[0]["value"] if options else None








    return options, sensor_values, options, cyl_value








@app.callback(
    Output("group-dropdown", "options"),
    Output("group-dropdown", "value"),
    Input("temp-dropdown", "value"),
    Input("iter-dropdown", "value")
)
def update_groups(temp, iter_files):
    if not temp or not iter_files:
        return [], None
    first = iter_files[0]
    df = load_data(temp, first)
    groups = categorize_sensors(df)
    options = [{"label": k, "value": k} for k, v in groups.items() if v]
    return options, (options[0]["value"] if options else None)








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
    if not group or not temp or not iter_files:
        return [], []
    triggered = ctx.triggered_id
    first = iter_files[0]
    df = load_data(temp, first)
    groups = categorize_sensors(df)
    options = [{"label": s, "value": s} for s in groups.get(group, [])]
    if triggered == "select-all-btn":
        values = groups[group]
    elif triggered == "clear-all-btn":
        values = []
    else:
        values = groups[group][:3]
    return options, values








@app.callback(
    Output("sensor-graph", "figure"),
    Input("sensor-dropdown", "value"),
    Input("group-dropdown", "value"),
    Input("temp-dropdown", "value"),
    Input("iter-dropdown", "value")
)
def update_graph(sensors, group, temp, iter_files):
    """
    If more than one iteration is selected, create a side-by-side row of subplots (one per run),
    with each run’s time‐axis and y‐axis autoscaled to that run’s data. Ensure equal spacing
    so that y-axis titles do not overlap adjacent plots. If only a single run, show a single plot.
    """
    fig = go.Figure()
    if not sensors or not temp or not iter_files:
        return fig.update_layout(title="No data selected")








    runs = iter_files
    n_runs = len(runs)








    if n_runs > 1:
        # Create horizontal subplots: one column per run; each subplot autoscale independently
        fig = make_subplots(
            rows=1,
            cols=n_runs,
            shared_xaxes=False,
            shared_yaxes=False,
            horizontal_spacing=0.12,  # Increase spacing so y-axis titles don't overlap
            subplot_titles=[f"{run}" for run in runs]
        )








        for i, run in enumerate(runs, start=1):
            df = load_data(temp, run)
            time_col = df.get("Scan Sweep Time (Sec)", df.index)
            for sensor in sensors:
                y = pd.to_numeric(df[sensor], errors="coerce")
                if sensor == "T111":
                    y = y * 0.025
                elif sensor == "T116" or sensor.startswith("T_ext"):
                    y = y * 0.05








                fig.add_trace(
                    go.Scatter(
                        x=time_col,
                        y=y,
                        mode="lines" if group == "Thermocouples" else "markers",
                        name=sensor,
                        showlegend=(i == 1)
                    ),
                    row=1,
                    col=i
                )








            # Force autoscaling on both axes; use automargin so titles won't overlap
            fig.update_xaxes(autorange=True, row=1, col=i, automargin=True, title_text="Time")
            fig.update_yaxes(
                autorange=True,
                row=1, col=i,
                automargin=True,
                title_text="Temperature (°C)" if group in ["Thermocouples", "Wall Temps"] else "Output (V)"
            )








        fig.update_layout(
            height=500,
            width=350 * n_runs,
            title=f"Sensor Comparison @ {temp}°C",
            showlegend=True,
            margin=dict(l=80, r=80, t=60, b=50)
        )
        return fig








    # Single‐run case:
    run = runs[0]
    df = load_data(temp, run)








    # If the user selected the special sensors T111 or T116 alone, show that specific plot:
    if sensors == ["T111"]:
        raw = pd.to_numeric(df["T111"], errors="coerce")
        conv = raw * 0.025
        fig.add_trace(go.Scatter(x=conv, y=raw, mode="markers", name="T111"))
        return fig.update_layout(
            title="Air Velocity (T111)",
            xaxis_title="Velocity (m/s)",
            yaxis_title="Output (V)",
            xaxis_autorange=True,
            yaxis_autorange=True,
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True)
        )








    if sensors == ["T116"]:
        raw = pd.to_numeric(df["T116"], errors="coerce")
        conv = raw * 0.05
        fig.add_trace(go.Scatter(x=conv, y=raw, mode="markers", name="T116"))
        return fig.update_layout(
            title="Air Temperature (T116)",
            xaxis_title="Temperature (°C)",
            yaxis_title="Output (V)",
            xaxis_autorange=True,
            yaxis_autorange=True,
            xaxis=dict(automargin=True),
            yaxis=dict(automargin=True)
        )








    # Otherwise, plot the selected sensors over time:
    time_col = df.get("Scan Sweep Time (Sec)", df.index)
    for sensor in sensors:
        y = pd.to_numeric(df[sensor], errors="coerce")
        fig.add_trace(go.Scatter(
            x=time_col,
            y=y,
            mode="lines" if group == "Thermocouples" else "markers",
            name=sensor
        ))








    y_label = "Temperature (°C)" if group in ["Thermocouples", "Wall Temps"] else "Output (V)"
    fig.update_layout(
        title=f"Sensor Readings – {temp}°C | {run}",
        xaxis_title="Time",
        yaxis_title=y_label,
        height=600,
        margin=dict(l=80, r=80, t=60, b=50),
        xaxis_autorange=True,
        yaxis_autorange=True,
        xaxis=dict(automargin=True),
        yaxis=dict(automargin=True)
    )








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
    # Load data and set slider/input bounds
    if not temp or not cyl_iter:
        return 0, 0, {}, 0, 0, 0, 0
    df = load_data(temp, cyl_iter)
    N = len(df)
    if N <= 1:
        return 0, 0, {0: "0"}, 0, 0, 0, 0
    step = max(1, N // 10)
    indices = list(range(0, N, step)) + [N - 1]
    unique_indices = sorted(set(indices))
    marks = {i: str(i) for i in unique_indices}
    min_val = 0
    max_val = N - 1
    # Determine which input triggered
    triggered = ctx.triggered_id if hasattr(ctx, 'triggered_id') else None
    # Clamp values
    if slider_val is None:
        slider_val = max_val
    if input_val is None:
        input_val = max_val
    slider_val = max(min_val, min(slider_val, max_val))
    input_val = max(min_val, min(input_val, max_val))
    # Sync logic
    if triggered == "time-slider":
        return min_val, max_val, marks, slider_val, min_val, max_val, slider_val
    elif triggered == "time-index-input":
        return min_val, max_val, marks, input_val, min_val, max_val, input_val
    else:
        return min_val, max_val, marks, max_val, min_val, max_val, max_val








@app.callback(
    Output("cylinder-3d", "figure"),
    Input("temp-dropdown", "value"),
    Input("cyl-iter-dropdown", "value"),
    Input("time-slider", "value"),
    Input("ring-select", "value")
)
def update_cylinder(temp, cyl_iter, time_idx, ring_select):
    """
    Rebuild the 3D cylinder‐rings plot using:
      • temp (to know folder),
      • cyl_iter (which file),
      • time_idx (which row in that CSV),
      • ring_select (list of radii to show; others transparent).
    """
    if not temp or not cyl_iter:
        return go.Figure().update_layout(title="No data")








    df = load_data(temp, cyl_iter)
    # Identify thermocouple columns T2–T29 specifically:
    temp_cols = [c for c in df.columns if re.fullmatch(r"T[2-9]$|T1[0-9]$|T2[0-9]$|T30$", c)]
    temp_cols = sorted(temp_cols, key=lambda x: int(x[1:]))








    # Clamp time_idx in case slider is out of range
    max_idx = len(df) - 1
    if time_idx < 0:
        time_idx = 0
    elif time_idx > max_idx:
        time_idx = max_idx








    fig = make_cylinder_surface(df, temp_cols, time_idx, ring_select)
    return fig








# --------------------------------------------------
# 6) RUN SERVER
# --------------------------------------------------








if __name__ == "__main__":
    app.run(debug=True)





