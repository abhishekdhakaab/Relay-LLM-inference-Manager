"""OpenTelemetry setup and trace-context helpers."""

from __future__ import annotations

import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_provider: Optional[TracerProvider] = None
_test_exporter: Optional[InMemorySpanExporter] = None


def configure_tracing(
    service_name: str = "llm-relay",
    otlp_endpoint: Optional[str] = None,
) -> TracerProvider:
    """Initialize one provider, exporting remotely only when configured."""
    global _provider, _test_exporter

    if _provider is not None:
        return _provider  # FastAPI test clients may create the app repeatedly.

    resource = Resource.create({"service.name": service_name})
    _provider = TracerProvider(resource=resource)

    endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        _provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        # In-memory spans keep tracing testable without a collector.
        _test_exporter = InMemorySpanExporter()
        _provider.add_span_processor(SimpleSpanProcessor(_test_exporter))

    trace.set_tracer_provider(_provider)
    return _provider


def get_tracer(name: str = "llm-relay") -> trace.Tracer:
    """Return a tracer from the already-configured provider."""
    return trace.get_tracer(name)


def get_test_exporter() -> Optional[InMemorySpanExporter]:
    """Expose the in-memory exporter when no remote endpoint is configured."""
    return _test_exporter


def extract_trace_id() -> str:
    """Return the active trace ID as 32 hex characters, or an empty string."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return ""
