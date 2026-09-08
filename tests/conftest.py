"""Keeps OpenTelemetry's console exporter quiet during the test run -- real
usage (CLI, API) still gets console spans by default."""
from agent_orchestrator.config import settings

settings.otel_console_export = False
