echo "=== Formatting ==="

# # Linters
echo ""
echo "Running Ruff"
poetry run ruff check --fix censorengine

# Check
echo ""
echo "=== Checking With Linter ==="
bash lint.sh