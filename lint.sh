
echo "Running Ruff"
uv run ruff check src/censor_engine --statistics

echo ""
echo "Running MyPy"
uv run mypy src/censor_engine 
