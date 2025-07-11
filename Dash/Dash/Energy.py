import os
import glob
import re
import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError
from CoolProp.CoolProp import PropsSI
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output


# ──────────────────────────────────────────────────────────────────────────────
# 1) Geometry & Constants
# ──────────────────────────────────────────────────────────────────────────────
PIPE_DIAMETER_M = 0.1082
PIPE_AREA       = np.pi * (PIPE_DIAMETER_M/2)**2
ATM_PRESSURE    = 101325    # Pa
PRESSURE_CONV   = 1.0       # ADC→Pa for P_diff channel


# ──────────────────────────────────────────────────────────────────────────────
# 2) Load all CSVs under data/<temperature>/ folders
# ──────────────────────────────────────────────────────────────────────────────
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


df_all = pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3) Preprocessing: Time, Mass Flow, ΔT
# ──────────────────────────────────────────────────────────────────────────────
df_all['Time'] = pd.to_datetime(df_all['Scan Sweep Time (Sec)'])


df_all['T_aire_C'] = df_all['116 (Vdc)- T_aire'] * 0.05
df_all['T_aire_K'] = df_all['T_aire_C'] + 273.15
df_all['V_aire']   = df_all['111 (Vdc)- V_aire'] * 0.025


df_all['rho_air']        = df_all['T_aire_K'].apply(
    lambda T: PropsSI('D','T',T,'P',ATM_PRESSURE,'Air')
)
df_all['mass_flow_rate'] = df_all['rho_air'] * PIPE_AREA * df_all['V_aire']


T1_col     = next(c for c in df_all.columns if c.strip().endswith('- T1'))
T30_col    = next(c for c in df_all.columns if c.strip().endswith('- T30'))
p_diff_col = next(c for c in df_all.columns if 'P_diff' in c)


df_all['T1_C']   = df_all[T1_col]
df_all['T30_C']  = df_all[T30_col]
df_all['deltaT'] = df_all['T30_C'] - df_all['T1_C']


# ──────────────────────────────────────────────────────────────────────────────
# 4) Compute Energy & Work Integrals
# ──────────────────────────────────────────────────────────────────────────────
df_all['mid_T_K'] = (df_all['T1_C'] + df_all['T30_C']) / 2 + 273.15
df_all['cp_air']  = df_all['mid_T_K'].apply(
    lambda T: PropsSI('Cpmass','T',T,'P',ATM_PRESSURE,'Air')
)


df_all['energy_rate'] = (
    df_all['mass_flow_rate']
  * df_all['cp_air']
  * (df_all['T1_C'] - df_all['T30_C'])
)


df_all['dP_Pa']     = df_all[p_diff_col] * PRESSURE_CONV
df_all['fan_power'] = PIPE_AREA * df_all['V_aire'] * df_all['dP_Pa']


df_all.sort_values(['Temperature','Iteration','Time'], inplace=True)
df_all['dt'] = df_all.groupby(['Temperature','Iteration'])['Time']\
                     .diff().dt.total_seconds().fillna(0)


df_all['dE_air'] = df_all['energy_rate'] * df_all['dt']
df_all['dW_fan'] = df_all['fan_power']  * df_all['dt']


df_all['E_air_cum'] = df_all.groupby(['Temperature','Iteration'])['dE_air'].cumsum()
df_all['W_fan_cum'] = df_all.groupby(['Temperature','Iteration'])['dW_fan'].cumsum()
df_all['E_st_cum']  = df_all['E_air_cum'] - df_all['W_fan_cum']


# ──────────────────────────────────────────────────────────────────────────────
# 5) Dash App & Layout
# ──────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("TES System Dashboard"),


    html.Div([
        html.Label("Temperature (°C)"),
        dcc.Dropdown(
            id='temp-dropdown',
            options=[{'label': f"{t}°C", 'value': t}
                     for t in sorted(df_all['Temperature'].unique())],
            value=sorted(df_all['Temperature'].unique())[0],
            clearable=False
        ),
    ], style={'width':'200px','display':'inline-block','marginRight':'20px'}),


    html.Div([
        html.Label("Iteration(s)"),
        dcc.Dropdown(id='iteration-dropdown', multi=True)
    ], style={'width':'300px','display':'inline-block'}),


    dcc.Graph(id='mass-flow-graph'),
    dcc.Graph(id='deltaT-graph'),
    dcc.Graph(id='energy-graph'),
    dcc.Graph(id='stored-energy-graph'),
    dcc.Graph(id='fan-work-graph'),
    dcc.Graph(id='efficiency-graph'),
])


# ──────────────────────────────────────────────────────────────────────────────
# Populate iteration-dropdown on temperature change
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('iteration-dropdown','options'),
    Output('iteration-dropdown','value'),
    Input('temp-dropdown','value')
)
def set_iterations(temp):
    its = sorted(df_all[df_all['Temperature']==temp]['Iteration'].unique(), key=int)
    opts = [{'label': f"Iter {i}", 'value': i} for i in its]
    return opts, [its[0]] if its else []


# ──────────────────────────────────────────────────────────────────────────────
# Helper to break axis-linking in facet plots
# ──────────────────────────────────────────────────────────────────────────────
def break_linking(fig):
    fig.for_each_xaxis(lambda ax: ax.update(matches=None))
    fig.for_each_yaxis(lambda ax: ax.update(matches=None))


# ──────────────────────────────────────────────────────────────────────────────
# Update all figures, using energy_rate jump to split phases
# ──────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output('mass-flow-graph','figure'),
    Output('deltaT-graph','figure'),
    Output('energy-graph','figure'),
    Output('stored-energy-graph','figure'),
    Output('fan-work-graph','figure'),
    Output('efficiency-graph','figure'),
    Input('temp-dropdown','value'),
    Input('iteration-dropdown','value'),
)
def update_all(temp, iterations):
    if not iterations:
        return ({},) * 6


    d = df_all[
        (df_all['Temperature']==temp) &
        (df_all['Iteration'].isin(iterations))
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


    # 3) Cumulative Air Energy with jump-based split
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


    fig3 = px.line(
        d2, x='Time', y='E_air_cum', color='phase',
        facet_col='Iteration', facet_col_wrap=2, markers=True,
        title=f"Cumulative Air Energy @ {temp}°C",
        labels={'E_air_cum':'Energy (J)','phase':'Phase'}
    )
    break_linking(fig3)


    # 4) Stored Energy
    fig4 = px.line(
        d, x='Time', y='E_st_cum',
        facet_col='Iteration', facet_col_wrap=2, markers=True,
        title=f"Stored Energy @ {temp}°C",
        labels={'E_st_cum':'Stored Energy (J)'}
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
        dd  = d[d['Iteration']==it]
        t_sw = switch_times[it]
        idx  = dd.index[dd['Time']==t_sw][0]


        E_ch  = dd.loc[idx,'E_air_cum']
        E_tot = dd['E_air_cum'].iat[-1]
        E_dch = E_tot - E_ch
        W_ch  = dd.loc[idx,'W_fan_cum']
        E_st  = E_ch - W_ch


        eta_rt  = 0.0 if E_ch == 0 else E_dch/E_ch
        eta_ch  = 0.0 if E_ch == 0 else E_st/E_ch
        eta_dch = 0.0 if E_st == 0 else E_dch/E_st


        rows += [
            {'Iteration':it,'Metric':'Round-trip',  'η':eta_rt},
            {'Iteration':it,'Metric':'Charging',    'η':eta_ch},
            {'Iteration':it,'Metric':'Discharging', 'η':eta_dch},
        ]


    df_eff = pd.DataFrame(rows)
    fig6 = px.bar(
        df_eff, x='Metric', y='η',
        facet_col='Iteration', facet_col_wrap=2,
        title=f"Efficiencies @ {temp}°C",
        labels={'η':'Efficiency'}
    )
    break_linking(fig6)


    return fig1, fig2, fig3, fig4, fig5, fig6


if __name__ == '__main__':
    app.run(debug=True)
#

