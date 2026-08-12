FROM python:3.11-slim

WORKDIR /app

# Install yt-dlp for YouTube transcript extraction
RUN pip install --no-cache-dir uv yt-dlp

# Install project dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8501

# Run ingestion pipeline (--no-prefect avoids ephemeral server bug) then start Streamlit
CMD ["bash", "-c", "uv run python -m ingestion.prefect_flow --no-prefect && uv run streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501"]
