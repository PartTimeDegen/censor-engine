echo "=== Formatting ==="

# # Linters
echo ""
echo "Running Ruff"
uv run ruff check . --fix
uv run ruff format .

# Check
echo ""
echo "=== Checking With Linter ==="
bash lint.sh