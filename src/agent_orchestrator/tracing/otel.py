"""OpenTelemetry wiring: one process-wide TracerProvider, exporting spans to
the console by default and to an OTLP collector when
`OTEL_EXPORTER_OTLP_ENDPOINT` is set -- so this plugs into real observability
tooling (Jaeger, Honeycomb, etc.) without code changes, while the trace
explorer / cost dashboards in this project are powered by the structured
`TraceStore` (SQLite) instead, since aggregating "cost per task type" or
"most expensive agent" isn't something span exporters do for you without a
metrics backend of their own.
"""
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from ..config import settings

_provider: TracerProvider | None = None


def _build_provider() -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: settings.otel_service_name})
    )
    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
            )
        )
    if settings.otel_console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    return provider


def get_tracer(name: str = "agent_orchestrator") -> trace.Tracer:
    global _provider
    if _provider is None:
        _provider = _build_provider()
        trace.set_tracer_provider(_provider)
    return trace.get_tracer(name)
