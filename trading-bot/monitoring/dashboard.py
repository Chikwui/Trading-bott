"""
Trading Bot Monitoring Dashboard

A real-time dashboard to monitor the trading bot's performance and system metrics.
"""
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os

# Configuration
METRICS_URL = "http://localhost:8000/metrics"
UPDATE_INTERVAL = 5  # seconds

# Initialize the Dash app with a dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Trading Bot Dashboard"

# App layout
app.layout = dbc.Container(
    fluid=True,
    children=[
        # Header
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.H1("Trading Bot Monitoring", className="text-center my-4"),
                    html.P("Real-time system metrics and performance monitoring", 
                          className="text-center text-muted mb-4"),
                ]),
                width=12,
            )
        ),
        
        # System Metrics Row
        dbc.Row(
            [
                # CPU Usage Gauge
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("CPU Usage", className="card-title"),
                            dcc.Graph(id="cpu-usage-gauge", style={"height": "300px"}),
                        ]),
                        className="mb-4"
                    ),
                    md=4
                ),
                
                # Memory Usage Gauge
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Memory Usage", className="card-title"),
                            dcc.Graph(id="memory-usage-gauge", style={"height": "300px"}),
                        ]),
                        className="mb-4"
                    ),
                    md=4
                ),
                
                # Disk Usage Gauge
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Disk Usage", className="card-title"),
                            dcc.Graph(id="disk-usage-gauge", style={"height": "300px"}),
                        ]),
                        className="mb-4"
                    ),
                    md=4
                ),
            ],
            className="mb-4",
        ),
        
        # Time Series Charts Row
        dbc.Row(
            [
                # Resource Usage Over Time
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H4("Resource Usage Over Time", className="card-title"),
                            dcc.Graph(id="resource-usage-chart", style={"height": "400px"}),
                        ]),
                        className="mb-4"
                    ),
                    md=12
                ),
            ]
        ),
        
        # Hidden div to store the metrics data
        dcc.Store(id='metrics-store'),
        
        # Interval component for auto-updating
        dcc.Interval(
            id="interval-component",
            interval=UPDATE_INTERVAL * 1000,  # in milliseconds
            n_intervals=0,
        ),
        
        # Footer
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.Hr(),
                    html.P(
                        [
                            "Monitoring Dashboard v1.0 | ",
                            html.Small(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "),
                            html.Small("Update every 5 seconds")
                        ],
                        className="text-center text-muted"
                    )
                ], className="mt-4"),
                width=12,
            )
        ),
    ],
    className="py-4",
)

# Store for historical metrics
metrics_history = {
    "timestamps": [],
    "cpu_usage": [],
    "memory_usage": [],
    "disk_usage": [],
    "request_count": [],
}

# Function to fetch and parse metrics
def fetch_metrics():
    try:
        response = requests.get(METRICS_URL, timeout=5)
        if response.status_code == 200:
            metrics = {}
            for line in response.text.split('\n'):
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics[parts[0]] = float(parts[1])
            return metrics
    except Exception as e:
        print(f"Error fetching metrics: {e}")
    return {}

# Callback to update all metrics
@app.callback(
    [
        Output("cpu-usage-gauge", "figure"),
        Output("memory-usage-gauge", "figure"),
        Output("disk-usage-gauge", "figure"),
        Output("resource-usage-chart", "figure"),
    ],
    [Input("interval-component", "n_intervals")],
)
def update_metrics(n):
    metrics = fetch_metrics()
    current_time = datetime.now()
    
    # Update history
    metrics_history["timestamps"].append(current_time)
    metrics_history["cpu_usage"].append(metrics.get("system_cpu_usage_percent", 0))
    metrics_history["memory_usage"].append(metrics.get("system_memory_usage_percent", 0))
    metrics_history["disk_usage"].append(metrics.get("system_disk_usage_percent", 0))
    metrics_history["request_count"].append(metrics.get("http_requests_total", 0))
    
    # Keep only the last 100 data points
    max_points = 100
    for key in metrics_history:
        metrics_history[key] = metrics_history[key][-max_points:]
    
    # Create gauge chart
    def create_gauge(value, title, min_val=0, max_val=100):
        return go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=value,
                title={"text": title},
                gauge={
                    "axis": {"range": [min_val, max_val]},
                    "bar": {"color": "#1f77b4"},
                    "steps": [
                        {"range": [0, 70], "color": "#2ecc71"},  # Green
                        {"range": [70, 90], "color": "#f39c12"},  # Orange
                        {"range": [90, 100], "color": "#e74c3c"},  # Red
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": value
                    },
                },
            )
        ).update_layout(
            margin=dict(t=50, b=10, l=30, r=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={"color": "white"}
        )
    
    # Create time series chart
    df = pd.DataFrame(metrics_history)
    if len(df) > 1:
        fig_ts = px.line(
            df,
            x="timestamps",
            y=["cpu_usage", "memory_usage", "disk_usage"],
            title="Resource Usage Over Time",
            labels={"value": "Usage %", "timestamps": "Time", "variable": "Metric"},
            color_discrete_map={
                "cpu_usage": "#3498db",
                "memory_usage": "#2ecc71",
                "disk_usage": "#9b59b6"
            }
        )
        
        fig_ts.update_layout(
            plot_bgcolor='rgba(0,0,0,0.2)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={"color": "white"},
            legend_title_text="",
            xaxis_title="",
            yaxis_title="Usage %",
            hovermode="x unified",
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1
            },
            margin=dict(t=50, b=50, l=50, r=50)
        )
    else:
        fig_ts = go.Figure()
        fig_ts.update_layout(
            title="Collecting data...",
            xaxis={"title": "Time"},
            yaxis={"title": "Usage %"},
            plot_bgcolor='rgba(0,0,0,0.2)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={"color": "white"},
        )
    
    return (
        create_gauge(metrics.get("system_cpu_usage_percent", 0), "CPU Usage"),
        create_gauge(metrics.get("system_memory_usage_percent", 0), "Memory Usage"),
        create_gauge(metrics.get("system_disk_usage_percent", 0), "Disk Usage"),
        fig_ts,
    )

if __name__ == "__main__":
    print("Starting Trading Bot Dashboard...")
    print(f"Dashboard will be available at: http://localhost:8050")
    print("Press Ctrl+C to stop the dashboard")
    app.run_server(debug=True, port=8050)
