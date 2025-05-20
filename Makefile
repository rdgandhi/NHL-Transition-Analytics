.PHONY: all data model app clean

all: data model app

data:
	python src/ingest.py

model:
	python src/models.py

app:
	streamlit run app/dashboard.py

clean:
	rm -rf data/*.parquet *.duckdb