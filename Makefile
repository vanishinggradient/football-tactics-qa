up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build

logs:
	docker-compose logs -f app

ingest:
	.venv/bin/python -m ingestion.prefect_flow

app:
	.venv/bin/streamlit run app/streamlit_app.py

clean:
	docker-compose down -v
