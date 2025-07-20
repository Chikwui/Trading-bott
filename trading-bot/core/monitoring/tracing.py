"""
Distributed tracing utilities for the trading bot.

This module provides OpenTelemetry-based distributed tracing
for monitoring and debugging the trading system.
"""
import contextlib
import logging
import os
from typing import Any, Dict, Optional, Callable, TypeVar, Type, cast

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, export
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.context import Context
from opentelemetry.propagate import set_global_textmap, get_global_textmap
from opentelemetry.propagators.composite import CompositeHTTPPropagator
from opentelemetry.trace.propagation import tracecontext

# Type variable for generic function wrapping
F = TypeVar('F', bound=Callable[..., Any])

logger = logging.getLogger(__name__)

class TracingConfig:
    """Configuration for distributed tracing."""
    
    def __init__(
        self,
        service_name: str = "trading-bot",
        service_version: str = "1.0.0",
        enable_console_exporter: bool = True,
        enable_jaeger: bool = False,
        jaeger_host: str = "localhost",
        jaeger_port: int = 6831,
        enable_otlp: bool = False,
        otlp_endpoint: Optional[str] = None,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.enable_console_exporter = enable_console_exporter
        self.enable_jaeger = enable_jaeger
        self.jaeger_host = jaeger_host
        self.jaeger_port = jaeger_port
        self.enable_otlp = enable_otlp
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")


def configure_tracing(config: Optional[TracingConfig] = None) -> None:
    """Configure OpenTelemetry tracing.
    
    Args:
        config: Tracing configuration. If None, uses environment variables.
    """
    if config is None:
        config = TracingConfig()
    
    # Configure resource
    resource = Resource.create({
        SERVICE_NAME: config.service_name,
        SERVICE_VERSION: config.service_version,
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Add span processors based on configuration
    if config.enable_console_exporter:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
    
    if config.enable_jaeger:
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=config.jaeger_host,
                agent_port=config.jaeger_port,
            )
            provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
        except ImportError:
            logger.warning("Jaeger exporter dependencies not installed. Run 'pip install opentelemetry-exporter-jaeger'")
    
    if config.enable_otlp and config.otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            otlp_exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        except ImportError:
            logger.warning("OTLP exporter dependencies not installed. Run 'pip install opentelemetry-exporter-otlp'")
    
    # Set the global tracer provider
    trace.set_tracer_provider(provider)
    
    # Configure context propagation
    propagator = CompositeHTTPPropagator([tracecontext.TraceContextTextMapPropagator()])
    set_global_textmap(propagator)


def get_tracer(name: Optional[str] = None) -> trace.Tracer:
    """Get a tracer instance.
    
    Args:
        name: Name of the tracer. If None, uses the module name of the caller.
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return trace.get_tracer(name)


def trace_function(tracer_name: Optional[str] = None, **span_attrs):
    """Decorator to trace function execution.
    
    Args:
        tracer_name: Name of the tracer to use. If None, uses the module name.
        **span_attrs: Additional attributes to set on the span.
    """
    def decorator(func: F) -> F:
        nonlocal tracer_name
        
        if tracer_name is None:
            import inspect
            frame = inspect.currentframe()
            while frame.f_back:
                frame = frame.f_back
            tracer_name = frame.f_globals.get('__name__', 'unknown')
        
        tracer = trace.get_tracer(tracer_name)
        
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    f"{func.__module__}.{func.__name__}",
                    attributes=span_attrs,
                    kind=trace.SpanKind.SERVER,
                ) as span:
                    try:
                        span.set_status(Status(StatusCode.OK))
                        return await func(*args, **kwargs)
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            return cast(F, async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with tracer.start_as_current_span(
                    f"{func.__module__}.{func.__name__}",
                    attributes=span_attrs,
                    kind=trace.SpanKind.SERVER,
                ) as span:
                    try:
                        span.set_status(Status(StatusCode.OK))
                        return func(*args, **kwargs)
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise
            
            return cast(F, sync_wrapper)
    
    return decorator


@contextlib.contextmanager
def start_span(name: str, tracer_name: Optional[str] = None, **attrs):
    """Context manager for creating a span.
    
    Args:
        name: Name of the span.
        tracer_name: Name of the tracer to use. If None, uses the module name.
        **attrs: Additional attributes to set on the span.
    """
    if tracer_name is None:
        import inspect
        frame = inspect.currentframe()
        while frame.f_back:
            frame = frame.f_back
        tracer_name = frame.f_globals.get('__name__', 'unknown')
    
    tracer = trace.get_tracer(tracer_name)
    with tracer.start_as_current_span(name, attributes=attrs) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def get_current_span() -> Optional[trace.Span]:
    """Get the current active span."""
    return trace.get_current_span()


def get_trace_context() -> Dict[str, str]:
    """Get the current trace context as a dictionary."""
    carrier: Dict[str, str] = {}
    get_global_textmap().inject(carrier)
    return carrier


def set_trace_context(carrier: Dict[str, str]) -> None:
    """Set the current trace context from a dictionary."""
    ctx = get_global_textmap().extract(carrier)
    trace.set_tracer_provider(trace.get_tracer_provider())
    trace.use_span(trace.get_current_span(ctx), end_on_exit=True)
