import coverage
import pytest

# Start coverage measurement
cov = coverage.Coverage()
cov.start()

# Run pytest
# Replace 'test_dir_or_file' with the directory or specific test file you want to test
pytest.main(['tests/'])

# Stop coverage measurement
cov.stop()
cov.save()

# Generate coverage report
cov.report()
