
echo "Running Ruff"
ruff check src/censor_engine

echo ""
echo "Running MyPy"
uv run mypy src/censor_engine 
