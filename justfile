# Cleans the repo.
clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo|build|generated$)" | xargs rm -rf
	@rm -rf src/*.egg-info/ build/ dist/ .coverage .pytest_cache/

# Applies formatting to all files.
format:
	isort --profile black .
	black .
	blacken-docs

# Lints all files.
lint:
	ruff check

# Runs all non-MPI tests and determines coverage.
test-cov workers="4":
	pytest -n {{workers}} --with-mpi --cov=src --cov-report=term --cov-report=xml tests/

# Runs all MPI-only tests with a given number of MPI ranks.
test-mpi ranks="3":
	mpiexec -np {{ranks}} pytest --only-mpi tests/

# Runs all tests.
test: test-mpi test-cov