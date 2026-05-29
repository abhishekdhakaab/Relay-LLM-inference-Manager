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
    """
    Initialise the global OTel TracerProvider once.

    If OTEL_EXPORTER_OTLP_ENDPOINT is set (or otlp_endpoint is provided),
    spans are exported to Jaeger via gRPC.  Otherwise (e.g. in unit tests
    or mock backend mode), an InMemorySpanExporter is used so tests can
    inspect spans without a running Jaeger instance.
    """
    global _provider, _test_exporter

    if _provider is not None:
        return _provider  # idempotent

    resource = Resource.create({"service.name": service_name})
    _provider = TracerProvider(resource=resource)

    endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")

    if endpoint:
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        _provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        # Fallback: in-memory exporter — useful in CI / mock mode
        _test_exporter = InMemorySpanExporter()
        _provider.add_span_processor(SimpleSpanProcessor(_test_exporter))

    trace.set_tracer_provider(_provider)
    return _provider


def get_tracer(name: str = "llm-relay") -> trace.Tracer:
    """Return a tracer from the already-configured provider."""
    return trace.get_tracer(name)


def get_test_exporter() -> Optional[InMemorySpanExporter]:
    """Expose the in-memory exporter for tests (None when real Jaeger is used)."""
    return _test_exporter


def extract_trace_id() -> str:
    """
    Return the current span's trace_id as a 32-char hex string.
    Returns empty string if there is no active span.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return ""
