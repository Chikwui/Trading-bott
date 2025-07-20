"""
Trading Dashboard

This module provides a web-based dashboard for monitoring trading activity,
including real-time metrics, position tracking, and alert management.
"""
import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Set, Union

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.wsgi import WSGIMiddleware
from starlette.middleware.cors import CORSMiddleware
from prometheus_client import start_http_server
from prometheus_async import aio

from core.utils.helpers import get_logger
from core.monitoring.monitoring_service import trade_monitor, METRICS
from core.utils.trade_logger import TradeEventType
from core.services.mt5_dashboard_service import MT5DashboardService
from core.monitoring.realtime import realtime_manager, publish_update

logger = get_logger(__name__)

# Initialize the Dash app
def create_dash_app():
    """Create and configure the Dash application."""
    global dash_app
    
    # Initialize Dash app
    dash_app = dash.Dash(
        __name__,
        server=False,  # We'll use FastAPI as the main server
        external_stylesheets=[dbc.themes.DARKLY],
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        suppress_callback_exceptions=True,
        assets_folder=static_dir
    )
    dash_app.title = "Trading Bot Dashboard"
    
    # Register the Dash app with FastAPI
    @fastapi_app.on_event("startup")
    async def startup_event():
        """Initialize the dashboard service on startup."""
        await realtime_manager.start()
    
    @fastapi_app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown."""
        await realtime_manager.close()
    
    return dash_app

# Create the Dash app
dash_app = create_dash_app()
app = dash_app.server  # WSGI app for Gunicorn compatibility

# Define colors
COLORS = {
    'background': '#222',
    'text': '#7FDBFF',
    'positive': '#2ecc71',
    'negative': '#e74c3c',
    'neutral': '#3498db',
    'grid': '#444'
}

# Global variables
dashboard_service = None
app = None
fastapi_app = FastAPI(title="Trading Bot Dashboard API")
dash_app = None

# Configure CORS
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
fastapi_app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Layout components
def build_header() -> html.Div:
    """Build the dashboard header."""
    return html.Div(
        className="app-header",
        children=[
            html.Div(
                className="app-header--title",
                children=[
                    html.H1("Trading Bot Dashboard", className="app-header--title-text"),
                    html.Div(
                        className="app-header--title-status",
                        id="connection-status",
                        children=[
                            html.Span("•", className="status-dot", id="status-dot"),
                            html.Span("Connected", className="status-text")
                        ]
                    )
                ]
            ),
            html.Div(
                className="app-header--controls",
                children=[
                    dcc.Interval(
                        id='interval-component',
                        interval=5*1000,  # in milliseconds
                        n_intervals=0
                    ),
                    html.Div(
                        className="last-updated",
                        children=[
                            html.Span("Last updated: "),
                            html.Span(id="last-updated", children=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        ]
                    )
                ]
            )
        ]
    )

def build_metrics_row(account_data: Dict = None) -> html.Div:
    """Build the metrics summary row."""
    if account_data is None:
        account_data = {}
        
    return html.Div(
        className="row metrics-row",
        children=[
            # Account Balance
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6("Account Balance", className="card-title"),
                        html.H3(
                            f"${account_data.get('balance', 0):,.2f}", 
                            className="card-text",
                            id="account-balance"
                        ),
                        html.Small("Available: ", className="text-muted"),
                        html.Small(
                            f"${account_data.get('free_margin', 0):,.2f}",
                            id="free-margin"
                        )
                    ]),
                    className="metric-card"
                ),
                md=3
            ),
            # Equity
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6("Equity", className="card-title"),
                        html.H3(
                            f"${account_data.get('equity', 0):,.2f}",
                            className="card-text",
                            id="account-equity"
                        ),
                        html.Small("Margin: ", className="text-muted"),
                        html.Small(
                            f"${account_data.get('margin', 0):,.2f}",
                            id="margin-used"
                        )
                    ]),
                    className="metric-card"
                ),
                md=3
            ),
            # P&L
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6("Today's P&L", className="card-title"),
                        html.H3(
                            "$0.00",
                            className="card-text",
                            id="daily-pnl"
                        ),
                        html.Small("MTD: ", className="text-muted"),
                        html.Small("$0.00", id="mtd-pnl")
                    ]),
                    className="metric-card"
                ),
                md=3
            ),
            # Open Positions
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H6("Open Positions", className="card-title"),
                        html.H3(
                            "0",
                            className="card-text",
                            id="open-positions"
                        ),
                        html.Small("Orders: ", className="text-muted"),
                        html.Small("0", id="open-orders")
                    ]),
                    className="metric-card"
                ),
                md=3
            )
        ]
    )

def build_positions_table(positions: List[Dict] = None) -> html.Div:
    """Build the positions table."""
    if positions is None:
        positions = []
        
    columns = [
        {"name": "Symbol", "id": "symbol"},
        {"name": "Type", "id": "type"},
        {"name": "Size", "id": "volume"},
        {"name": "Entry", "id": "open_price"},
        {"name": "Current", "id": "current_price"},
        {"name": "P&L", "id": "profit"},
        {"name": "Actions", "id": "actions"}
    ]
    
    return html.Div(
        className="card mt-3",
        children=[
            dbc.CardHeader("Open Positions", className="card-header"),
            dbc.CardBody(
                dash_table.DataTable(
                    id='positions-table',
                    columns=columns,
                    data=positions,
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'textAlign': 'left',
                        'padding': '8px',
                        'color': COLORS['text']
                    },
                    style_header={
                        'backgroundColor': 'rgb(30, 30, 30)',
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(50, 50, 50)',
                        },
                        {
                            'if': {'row_index': 'even'},
                            'backgroundColor': 'rgb(40, 40, 40)',
                        },
                        {
                            'if': {
                                'filter_query': '{profit} > 0',
                                'column_id': 'profit'
                            },
                            'color': COLORS['positive']
                        },
                        {
                            'if': {
                                'filter_query': '{profit} < 0',
                                'column_id': 'profit'
                            },
                            'color': COLORS['negative']
                        }
                    ]
                )
            )
        ]
    )

# Real-time data endpoints
@fastapi_app.get("/api/stream")
async def stream_data(request: Request):
    """Stream real-time data updates."""
    # Subscribe to all relevant channels
    channels = [
        'account_update',
        'positions_update',
        'orders_update',
        'trade_event',
        'metrics_update'
    ]
    
    return StreamingResponse(
        realtime_manager.stream_response(request, channels),
        media_type="text/event-stream"
    )

# Mount Dash app under FastAPI
fastapi_app.mount("/", WSGIMiddleware(dash_app.server))

# App layout
dash_app.layout = html.Div(
    [
        # Hidden div to store real-time data
        dcc.Store(id='realtime-store', data={}),
        # SSE client for real-time updates
        dcc.Interval(id='sse-interval', interval=1000, max_intervals=1),  # Trigger once on load
        html.Div(id='sse-output', style={'display': 'none'}),
        
        # Main container
        dbc.Container(
            fluid=True,
            className="dashboard-container",
            style={
                'backgroundColor': COLORS['background'],
                'color': 'white',
                'minHeight': '100vh',
                'padding': '20px'
            },
            children=[
                # Header
                build_header(),
                
                # Main content
                dbc.Row([
                    # Left sidebar
                    dbc.Col(
                        [
                            html.H4("Account Overview", className="mb-3"),
                            html.Div(id="account-metrics"),
                            html.Hr(),
                            html.H4("Positions", className="mb-3"),
                            dash_table.DataTable(
                                id='positions-table',
                                columns=[
                                    {"name": "Symbol", "id": "symbol"},
                                    {"name": "Volume", "id": "volume"},
                                    {"name": "Price", "id": "price"},
                                    {"name": "Profit", "id": "profit"},
                                    {"name": "Time", "id": "time"}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '8px',
                                    'color': COLORS['text'],
                                    'backgroundColor': COLORS['background']
                                },
                                style_header={
                                    'backgroundColor': '#333',
                                    'fontWeight': 'bold'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(50, 50, 50)'
                                    }
                                ]
                            ),
                            html.Hr(),
                            html.H4("Orders", className="mb-3"),
                            dash_table.DataTable(
                                id='orders-table',
                                columns=[
                                    {"name": "Symbol", "id": "symbol"},
                                    {"name": "Type", "id": "type"},
                                    {"name": "Volume", "id": "volume"},
                                    {"name": "Price", "id": "price"},
                                    {"name": "Time", "id": "time"}
                                ],
                                style_table={'overflowX': 'auto'},
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '8px',
                                    'color': COLORS['text'],
                                    'backgroundColor': COLORS['background']
                                },
                                style_header={
                                    'backgroundColor': '#333',
                                    'fontWeight': 'bold'
                                },
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(50, 50, 50)'
                                    }
                                ]
                            )
                        ],
                        md=4
                    ),
                    
                    # Main content area
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dcc.Graph(
                                        id='pnl-chart', 
                                        className='chart-container',
                                        style={'height': '400px'}
                                    ),
                                    dcc.Graph(
                                        id='equity-chart', 
                                        className='chart-container',
                                        style={'height': '400px', 'marginTop': '20px'}
                                    )
                                ]
                            )
                        ],
                        md=8
                    )
                ])
            ]
        )
    ]
)

# Callbacks
@app.callback(
    Output('last-updated', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_last_updated(n):
    """Update the last updated timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@app.callback(
    [
        Output('account-data-store', 'data'),
        Output('positions-store', 'data'),
        Output('orders-store', 'data')
    ],
    [Input('interval-component', 'n_intervals')]
)
def update_data(n):
    """Update account, positions, and orders data."""
    if dashboard_service is None:
        return {}, [], []
        
    # In a real implementation, we would fetch this data from the dashboard service
    # For now, return empty data
    return {}, [], []

@app.callback(
    Output('metrics-row', 'children'),
    [Input('account-data-store', 'data')]
)
def update_metrics(account_data):
    """Update the metrics row with account data."""
    return build_metrics_row(account_data)

@app.callback(
    Output('positions-container', 'children'),
    [Input('positions-store', 'data')]
)
def update_positions(positions):
    """Update the positions table."""
    return build_positions_table(positions)

def init_dashboard(mt5_broker, mt5_provider, host: str = '0.0.0.0', port: int = 8050, debug: bool = False):
    """Initialize and run the dashboard.
    
    Args:
        mt5_broker: MT5 broker instance
        mt5_provider: MT5 data provider instance
        host: Host to run the dashboard on
        port: Port to run the dashboard on
        debug: Whether to run in debug mode
    """
    global dashboard_service, dash_app
    
    try:
        # Initialize dashboard service
        dashboard_service = MT5DashboardService(mt5_broker, mt5_provider)
        
        # Start Prometheus metrics server
        start_http_server(8000)
        
        # Register callbacks
        register_callbacks(dash_app, dashboard_service)
        
        logger.info(f"Starting dashboard on http://{host}:{port}")
        
        if debug:
            # In debug mode, run with auto-reload
            uvicorn.run(
                "core.monitoring.dashboard:fastapi_app",
                host=host,
                port=port,
                reload=True,
                log_level="debug" if debug else "info"
            )
        else:
            # In production, use multiple workers
            uvicorn.run(
                "core.monitoring.dashboard:fastapi_app",
                host=host,
                port=port,
                workers=4,
                log_level="info"
            )
            
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        raise

def register_callbacks(dash_app, dashboard_service):
    """Register all Dash callbacks."""
    
    @dash_app.callback(
        Output('realtime-store', 'data'),
        [Input('sse-interval', 'n_intervals')],
        prevent_initial_call=False
    )
    def initialize_sse(n):
        """Initialize the SSE connection."""
        return {}
    
    @dash_app.callback(
        Output('last-updated', 'children'),
        [Input('realtime-store', 'data')]
    )
    def update_last_updated(data):
        """Update the last updated timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @dash_app.callback(
        [
            Output('account-metrics', 'children'),
            Output('positions-table', 'data'),
            Output('orders-table', 'data'),
            Output('pnl-chart', 'figure'),
            Output('equity-chart', 'figure'),
        ],
        [Input('realtime-store', 'data')]
    )
    def update_dashboard(data):
        """Update the dashboard with new data."""
        # This will be triggered whenever realtime-store is updated by the SSE client
        # The actual data is passed through the dcc.Store component
        
        # Get the latest data from the store
        account_data = data.get('account', {})
        positions = data.get('positions', [])
        orders = data.get('orders', [])
        
        # Update metrics
        metrics = build_metrics_row(account_data)
        
        # Update tables
        positions_table = [
            {
                'symbol': p.get('symbol', ''),
                'volume': p.get('volume', 0),
                'price': p.get('price', 0),
                'profit': p.get('profit', 0),
                'swap': p.get('swap', 0),
                'commission': p.get('commission', 0),
                'time': p.get('time', '')
            }
            for p in positions
        ]
        
        orders_table = [
            {
                'ticket': o.get('ticket', 0),
                'symbol': o.get('symbol', ''),
                'type': o.get('type', ''),
                'volume': o.get('volume', 0),
                'price': o.get('price', 0),
                'sl': o.get('sl', 0),
                'tp': o.get('tp', 0),
                'time': o.get('time', '')
            }
            for o in orders
        ]
        
        # Update charts (simplified)
        pnl_chart = go.Figure()
        equity_chart = go.Figure()
        
        # Add some sample data (replace with real data)
        if account_data:
            pnl_chart.add_trace(go.Indicator(
                mode="number+delta",
                value=account_data.get('balance', 0),
                delta={'reference': account_data.get('prev_balance', 0), 'relative': True},
                title={"text": "Balance"}
            ))
            
            equity_chart.add_trace(go.Indicator(
                mode="number+delta",
                value=account_data.get('equity', 0),
                delta={'reference': account_data.get('prev_equity', 0), 'relative': True},
                title={"text": "Equity"}
            ))
        
        return metrics, positions_table, orders_table, pnl_chart, equity_chart
    
    @dash_app.callback(
        Output('connection-status', 'children'),
        [Input('realtime-store', 'data')]
    )
    def update_connection_status(data):
        """Update the connection status indicator."""
        is_connected = data.get('is_connected', False)
        
        return [
            html.Span(
                "•",
                className=f"status-dot {'connected' if is_connected else 'disconnected'}",
                id="status-dot"
            ),
            html.Span(
                "Connected" if is_connected else "Disconnected",
                className="status-text"
            )
        ]
    
    # Register additional callbacks here
    # ...

# Client-side JavaScript for SSE
sse_js = """
<script>
// Initialize SSE connection
function initSSE() {
    const eventSource = new EventSource('/api/stream');
    
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        console.log('Received update:', data);
        
        // Update the store with new data
        const store = document.getElementById('realtime-store');
        const currentData = JSON.parse(store.getAttribute('data') || '{}');
        
        // Merge the new data with existing data
        const newData = {
            ...currentData,
            [data.channel]: data.data,
            last_updated: new Date().toISOString()
        };
        
        // Update the store
        store.setAttribute('data', JSON.stringify(newData));
        
        // Trigger a Dash callback
        store.dispatchEvent(new Event('change'));
    };
    
    eventSource.onerror = function(error) {
        console.error('SSE error:', error);
        
        // Attempt to reconnect after a delay
        eventSource.close();
        setTimeout(initSSE, 5000);
    };
    
    return eventSource;
}

// Initialize when the page loads
document.addEventListener('DOMContentLoaded', initSSE);
</script>
"""

# Add the SSE client to the Dash app
dash_app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Add custom styles here */
            .status-dot {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 5px;
            }
            .status-dot.connected {
                background-color: #2ecc71;
            }
            .status-dot.disconnected {
                background-color: #e74c3c;
            }
            .status-text {
                font-size: 14px;
                color: #7f8c8d;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
            ''' + sse_js + '''
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    # For testing without MT5
    init_dashboard(None, None, debug=True)
