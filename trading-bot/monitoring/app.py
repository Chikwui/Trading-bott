"""
Trading System Monitoring Dashboard

This module provides a web-based dashboard for monitoring the trading system's
performance, circuit breakers, and event processing metrics.
"""
import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from .config import config
from .metrics import metrics, CIRCUIT_BREAKER_STATE, EVENT_PROCESSED_TOTAL, EVENT_QUEUE_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=config.DASHBOARD_TITLE,
    description="Real-time monitoring dashboard for the trading system",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

# Initialize templates
templates = Jinja2Templates(directory=config.TEMPLATES_DIR)

# Security
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    """Basic authentication for the dashboard."""
    if not config.AUTH_ENABLED:
        return True
        
    correct_username = config.AUTH_USERNAME
    correct_password = config.AUTH_PASSWORD
    
    if (credentials.username != correct_username or 
            credentials.password != correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

# Initialize Dash app
dash_app = Dash(
    __name__,
    server=app,
    url_base_pathname="/dashboard/",
    external_stylesheets=[getattr(dbc.themes, config.DASHBOARD_THEME)],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# Define dashboard layout
dash_app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H1(config.DASHBOARD_TITLE, className="display-4"),
                html.P("Real-time system monitoring and metrics", className="text-muted"),
                html.Hr(),
            ]),
            className="mb-4"
        )
    ]),
    
    # System Status Row
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("System Status", className="h5"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(id='system-status-gauge', config={'staticPlot': False}),
                            width=4
                        ),
                        dbc.Col([
                            dbc.Row([
                                dbc.Col(
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("Uptime", className="card-title"),
                                            html.H3(id="uptime-display", className="card-text"),
                                        ])
                                    ], className="mb-3"),
                                    width=6
                                ),
                                dbc.Col(
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("CPU Usage", className="card-title"),
                                            html.H3(id="cpu-usage-display", className="card-text"),
                                        ])
                                    ], className="mb-3"),
                                    width=6
                                ),
                            ]),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H4("Memory Usage", className="card-title"),
                                            html.H3(id="memory-usage-display", className="card-text"),
                                        ])
                                    ]),
                                    width=12
                                ),
                            ]),
                        ], width=8),
                    ]),
                ]),
            ], className="mb-4"),
            width=12
        )
    ]),
    
    # Circuit Breakers Row
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Circuit Breakers", className="h5"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(id='cb-status-gauge'),
                            width=4
                        ),
                        dbc.Col(
                            dcc.Graph(id='cb-metrics-chart'),
                            width=8
                        )
                    ])
                ])
            ], className="mb-4"),
            width=12
        )
    ]),
    
    # Event Processing Row
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Event Processing", className="h5"),
                dbc.CardBody([
                    dcc.Graph(id='event-metrics-chart')
                ])
            ], className="mb-4"),
            width=12
        )
    ]),
    
    # Queue Size Row
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Event Queue", className="h5"),
                dbc.CardBody([
                    dcc.Graph(id='queue-size-chart')
                ])
            ]),
            width=12
        )
    ]),
    
    # Hidden div for storing data
    dcc.Store(id='metrics-store'),
    
    # Auto-refresh component
    dcc.Interval(
        id='interval-component',
        interval=config.REFRESH_INTERVAL,
        n_intervals=0
    )
], fluid=True, className="py-4")

# Callbacks for updating dashboard components
@dash_app.callback(
    [
        Output('system-status-gauge', 'figure'),
        Output('uptime-display', 'children'),
        Output('cpu-usage-display', 'children'),
        Output('memory-usage-display', 'children'),
        Output('cb-status-gauge', 'figure'),
        Output('cb-metrics-chart', 'figure'),
        Output('event-metrics-chart', 'figure'),
        Output('queue-size-chart', 'figure'),
    ],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update all dashboard components."""
    # Update system metrics
    metrics.update_system_metrics()
    
    # Get current metrics
    cb_states = metrics.get_metric("circuit_breaker_state")._samples()
    cb_transitions = metrics.get_metric("circuit_breaker_transitions_total")._samples()
    event_metrics = metrics.get_metric("event_processed_total")._samples()
    queue_size = metrics.get_metric("event_queue_size")._value
    
    # System status gauge
    system_status_fig = create_system_status_gauge()
    
    # System metrics
    uptime = metrics.get_metric("system_uptime_seconds")._value
    cpu_usage = metrics.get_metric("system_cpu_usage_percent")._value
    memory_usage = metrics.get_metric("system_memory_usage_bytes")._samples()
    
    # Format system metrics for display
    uptime_str = str(timedelta(seconds=int(uptime)))
    cpu_str = f"{cpu_usage:.1f}%"
    
    # Get memory usage (RSS)
    mem_rss = next((s for s in memory_usage if s.labels["type"] == "rss"), None)
    mem_str = f"{mem_rss.value / (1024**2):.1f} MB" if mem_rss else "N/A"
    
    # Circuit breaker status
    cb_status_fig = create_circuit_breaker_gauge(cb_states)
    
    # Circuit breaker metrics
    cb_metrics_fig = create_circuit_breaker_metrics(cb_transitions)
    
    # Event metrics
    event_metrics_fig = create_event_metrics_chart(event_metrics)
    
    # Queue size chart
    queue_fig = create_queue_size_chart()
    
    return (
        system_status_fig,
        uptime_str,
        cpu_str,
        mem_str,
        cb_status_fig,
        cb_metrics_fig,
        event_metrics_fig,
        queue_fig,
    )

def create_system_status_gauge():
    """Create system status gauge."""
    # This is a simplified example - in a real app, you'd check various system metrics
    return {
        'data': [
            go.Indicator(
                mode="gauge+number",
                value=95,  # Example value
                title={'text': "System Health"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 80], 'color': "lightgreen"},
                        {'range': [80, 95], 'color': "yellow"},
                        {'range': [95, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            )
        ],
        'layout': {
            'height': 250,
            'margin': {'l': 20, 'r': 20, 't': 50, 'b': 20},
        }
    }

def create_circuit_breaker_gauge(cb_states):
    """Create circuit breaker status gauge."""
    # Count open/closed circuit breakers
    closed = sum(1 for s in cb_states if s.value == 0)
    total = len(cb_states) or 1  # Avoid division by zero
    
    return {
        'data': [
            go.Indicator(
                mode="gauge+number",
                value=closed,
                title={'text': f"Circuit Breakers Closed"},
                gauge={
                    'axis': {'range': [0, total]},
                    'bar': {'color': "green" if closed == total else "red"},
                    'steps': [
                        {'range': [0, total], 'color': "lightgray"},
                    ],
                }
            )
        ],
        'layout': {
            'height': 250,
            'margin': {'l': 20, 'r': 20, 't': 50, 'b': 20},
        }
    }

def create_circuit_breaker_metrics(transitions):
    """Create circuit breaker metrics chart."""
    # Process transitions data
    # This is a simplified example - in a real app, you'd process the actual metrics
    
    return {
        'data': [
            go.Bar(
                name='Closed to Open',
                x=['Order Execution', 'Market Data', 'Risk Engine'],
                y=[5, 3, 2],  # Example data
                marker_color='indianred'
            ),
            go.Bar(
                name='Open to Half-Open',
                x=['Order Execution', 'Market Data', 'Risk Engine'],
                y=[3, 2, 1],  # Example data
                marker_color='lightblue'
            ),
            go.Bar(
                name='Half-Open to Closed',
                x=['Order Execution', 'Market Data', 'Risk Engine'],
                y=[7, 5, 4],  # Example data
                marker_color='lightgreen'
            )
        ],
        'layout': {
            'title': 'Circuit Breaker State Transitions',
            'barmode': 'group',
            'legend': {'orientation': 'h', 'y': 1.1},
            'margin': {'l': 50, 'r': 20, 't': 50, 'b': 50},
            'height': 300,
        }
    }

def create_event_metrics_chart(events):
    """Create event processing metrics chart."""
    # This is a simplified example - in a real app, you'd process the actual metrics
    
    return {
        'data': [
            go.Bar(
                name='Success',
                x=['Order Created', 'Market Update', 'Position Update'],
                y=[1200, 124500, 2400],  # Example data
                marker_color='lightgreen'
            ),
            go.Bar(
                name='Error',
                x=['Order Created', 'Market Update', 'Position Update'],
                y=[12, 125, 48],  # Example data
                marker_color='indianred'
            )
        ],
        'layout': {
            'title': 'Event Processing Metrics',
            'barmode': 'stack',
            'legend': {'orientation': 'h', 'y': 1.1},
            'margin': {'l': 50, 'r': 20, 't': 50, 'b': 50},
            'height': 350,
        }
    }

def create_queue_size_chart():
    """Create queue size chart."""
    # This is a simplified example - in a real app, you'd use real queue metrics
    
    # Generate sample data for the last hour
    now = datetime.now()
    times = [(now - timedelta(minutes=i)).strftime('%H:%M') for i in range(60, -1, -5)]
    queue_sizes = [max(0, 100 + i * 2) for i in range(len(times))]
    
    return {
        'data': [
            go.Scatter(
                x=times,
                y=queue_sizes,
                mode='lines+markers',
                name='Queue Size',
                line=dict(color='royalblue', width=2),
                marker=dict(size=6)
            )
        ],
        'layout': {
            'title': 'Event Queue Size (Last Hour)',
            'xaxis': {'title': 'Time'},
            'yaxis': {'title': 'Queue Size'},
            'margin': {'l': 50, 'r': 20, 't': 50, 'b': 50},
            'height': 300,
            'shapes': [
                # Warning threshold
                {
                    'type': 'line',
                    'x0': 0,
                    'y0': 1000,
                    'x1': 1,
                    'y1': 1000,
                    'xref': 'paper',
                    'yref': 'y',
                    'line': {
                        'color': 'orange',
                        'width': 2,
                        'dash': 'dash',
                    },
                },
                # Critical threshold
                {
                    'type': 'line',
                    'x0': 0,
                    'y0': 5000,
                    'x1': 1,
                    'y1': 5000,
                    'xref': 'paper',
                    'yref': 'y',
                    'line': {
                        'color': 'red',
                        'width': 2,
                        'dash': 'dash',
                    },
                },
            ],
            'annotations': [
                {
                    'x': 0.02,
                    'y': 1000,
                    'xref': 'paper',
                    'yref': 'y',
                    'text': 'Warning',
                    'showarrow': False,
                    'yshift': 10,
                    'font': {'color': 'orange'}
                },
                {
                    'x': 0.02,
                    'y': 5000,
                    'xref': 'paper',
                    'yref': 'y',
                    'text': 'Critical',
                    'showarrow': False,
                    'yshift': 10,
                    'font': {'color': 'red'}
                },
            ]
        }
    }

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/metrics")
async def get_metrics():
    """Get all metrics in Prometheus format."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

def run_server():
    """Run the monitoring server."""
    # Ensure required directories exist
    config.STATIC_DIR.mkdir(parents=True, exist_ok=True)
    config.TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a simple index.html if it doesn't exist
    index_file = config.TEMPLATES_DIR / "index.html"
    if not index_file.exists():
        index_file.write_text("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading System Monitor</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .dashboard-container { padding: 20px; }
                .dashboard-header { margin-bottom: 30px; }
                iframe { width: 100%; height: 100vh; border: none; }
            </style>
        </head>
        <body>
            <div class="container-fluid dashboard-container">
                <div class="row dashboard-header">
                    <div class="col-12">
                        <h1>Trading System Monitor</h1>
                        <p class="lead">Real-time monitoring dashboard</p>
                        <hr>
                    </div>
                </div>
                <div class="row">
                    <div class="col-12">
                        <iframe src="/dashboard/"></iframe>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """)
    
    # Start the metrics server in a separate thread
    metrics_server = metrics.start_metrics_server(port=8001)
    
    try:
        # Start the FastAPI server
        uvicorn.run(
            app,
            host=config.HOST,
            port=config.PORT,
            reload=config.RELOAD,
            debug=config.DEBUG,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down monitoring server...")
    finally:
        # Cleanup if needed
        pass

if __name__ == "__main__":
    run_server()
