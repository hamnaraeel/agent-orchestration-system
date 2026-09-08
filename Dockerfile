FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY ui ./ui
COPY demo.py ./

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1

# One image, many roles -- docker-compose.yml overrides `command:` per
# service (api / celery worker / each Streamlit UI / the demo script).
CMD ["python", "-m", "agent_orchestrator.run", "--help"]
