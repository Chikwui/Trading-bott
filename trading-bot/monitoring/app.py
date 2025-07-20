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
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import logging
import sys
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from monitoring.config import config, security_config
from monitoring.metrics import metrics
from monitoring.security import limiter, api_key_required, get_api_key, verify_api_key
from monitoring.trading_metrics import trading_metrics, TradeDirection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('monitoring.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=config.DASHBOARD_TITLE,
    description="Trading System Monitoring Dashboard",
    version="0.1.0",
    docs_url="/docs" if config.DEBUG else None,
    redoc_url="/redoc" if config.DEBUG else None,
    openapi_url="/openapi.json" if config.DEBUG else None
)

# Security middleware
if security_config.ENABLE_AUTH:
    from monitoring.security import setup_security
    setup_security(app)

# Add other middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses > 1KB

# Security headers middleware is added in security.setup_security()

# Configure CORS with security settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in security_config.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)

# Trusted hosts middleware
if not config.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],  # In production, replace with your domain
    )
    
    # Force HTTPS in production
    app.add_middleware(HTTPSRedirectMiddleware)

# Mount static files with cache control
def set_cache_control_header(request, response):
    """Set cache control headers for static files."""
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "public, max-age=31536000"  # 1 year
    return response

app.mount(
    "/static", 
    StaticFiles(directory=config.STATIC_DIR), 
    name="static"
)
app.middleware("http")(set_cache_control_header)

# Templates
templates = Jinja2Templates(directory=config.TEMPLATES_DIR)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    metrics.record_event("request_processed", {"path": request.url.path, "method": request.method})
    return response

# Routes
@app.get("/", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def dashboard(
    request: Request,
    api_key: str = Security(api_key_required)
):
    """Render the main dashboard."""
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": config.DASHBOARD_TITLE,
            "refresh_interval": config.REFRESH_INTERVAL,
            "auth_enabled": security_config.ENABLE_AUTH
        }
    )

@app.get("/metrics")
@limiter.limit("10/minute")
async def get_metrics(
    request: Request,
    api_key: str = Security(api_key_required)
):
    """Expose Prometheus metrics with authentication."""
    return metrics.get_metrics_response()

@app.get("/api/health")
async def health_check():
    """Health check endpoint with basic system status."""
    from psutil import cpu_percent, virtual_memory
    
    return {
        "status": "ok",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "cpu_percent": cpu_percent(),
            "memory_percent": virtual_memory().percent,
            "uptime_seconds": metrics.get_metric("system_uptime_seconds").collect()[0].samples[0].value
        },
        "trading": {
            "active_trades": len(trading_metrics.open_trades),
            "total_trades": len(trading_metrics.trades),
            "metrics_available": True
        }
    }

# Trading metrics endpoints
@app.get("/api/trading/metrics/summary")
@limiter.limit("30/minute")
async def get_trading_metrics_summary(
    request: Request,
    api_key: str = Security(api_key_required)
):
    """Get summary of trading metrics."""
    try:
        return {
            "status": "success",
            "data": trading_metrics.get_summary_metrics()
        }
    except Exception as e:
        logger.error(f"Error getting trading metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving trading metrics"
        )

@app.get("/api/trading/trades")
@limiter.limit("30/minute")
async def get_trade_history(
    request: Request,
    symbol: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    api_key: str = Security(api_key_required)
):
    """Get trade history with optional filtering."""
    try:
        trades = list(trading_metrics.trades.values())
        
        # Apply filters
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
            
        # Sort by exit time (most recent first)
        trades.sort(key=lambda t: t.exit_time, reverse=True)
        
        # Apply pagination
        paginated_trades = trades[offset:offset + limit]
        
        return {
            "status": "success",
            "data": [vars(trade) for trade in paginated_trades],
            "pagination": {
                "total": len(trades),
                "limit": limit,
                "offset": offset
            }
        }
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving trade history"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with JSON responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with JSON responses."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"status": "error", "detail": "Internal server error"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    # Initialize trading metrics with default starting equity
    trading_metrics.initialize_equity(10000.0)  # Default starting equity
    logger.info("Trading metrics initialized")

if __name__ == "__main__":
    # In production, use a proper ASGI server like uvicorn with multiple workers
    uvicorn.run(
        "monitoring.app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level="info" if not config.DEBUG else "debug",
        proxy_headers=True,
        forwarded_allow_ips="*"
    )

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
