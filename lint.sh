
echo "Running Ruff"
poetry run ruff check src/censorengine

echo ""
echo "Running MyPy"
poetry run mypy src/censorengine 
