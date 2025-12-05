# ZEONE Testing Guide

[QA] Comprehensive guide for running and writing tests.

## Overview

ZEONE uses `pytest` with `pytest-asyncio` for testing. Tests are organized into three categories:

| Category | Path | Purpose | Speed |
|----------|------|---------|-------|
| Unit | `tests/unit/` | Isolated logic tests | Fast (<1s) |
| Integration | `tests/integration/` | Module interaction | Medium (1-10s) |
| E2E | `tests/e2e/` | Full scenarios | Slow (10s+) |

## Quick Start

```bash
# Install test dependencies
pip install -r requirements/dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run only unit tests (fast)
pytest tests/unit/ -v

# Run specific module
pytest tests/unit/core/test_crypto.py -v
```

## Test Categories

### Unit Tests (`tests/unit/`)

Fast, isolated tests without I/O. No network, no database writes.

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with markers
pytest -m unit -v
```

Structure:
```
tests/unit/
├── core/
│   ├── test_crypto.py      # Cryptography
│   ├── test_wire.py        # Binary protocol
│   └── test_merkle.py      # Merkle tree
├── economy/
│   ├── test_ledger.py      # Ledger operations
│   └── test_trust.py       # Trust score math
└── agents/
    └── test_manager.py     # Agent manager
```

### Integration Tests (`tests/integration/`)

Tests with real async I/O, temporary databases.

```bash
# Run integration tests
pytest tests/integration/ -v

# Skip slow tests
pytest tests/integration/ -v -m "not slow"
```

Structure:
```
tests/integration/
├── vpn/
│   └── test_socks.py       # SOCKS proxy
├── dht/
│   └── test_kademlia.py    # DHT operations
└── chain/
    └── test_settlement.py  # Blockchain
```

### E2E Tests (`tests/e2e/`)

Full user scenarios with multiple nodes.

```bash
# Run E2E tests (slow)
pytest tests/e2e/ -v --timeout=120
```

### Stress Tests

Load testing (run separately from CI):

```bash
# Run stress test
python tests/stress_test.py --clients 100 --duration 60

# Via pytest
pytest tests/stress_test.py -v -m stress
```

## Coverage

### Running with Coverage

```bash
# Generate HTML report
pytest --cov=. --cov-config=.coveragerc --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Console report
pytest --cov=. --cov-report=term-missing
```

### Coverage Configuration

Coverage is configured in `.coveragerc`:

- **Minimum threshold**: 80% (CI fails below this)
- **Excluded paths**: tests/, scripts/, webui/, contracts/
- **Branch coverage**: Enabled

### Interpreting HTML Report

1. **Green lines**: Covered code
2. **Red lines**: Not covered
3. **Yellow lines**: Partial branch coverage
4. **Gray lines**: Excluded (pragma: no cover)

Focus on:
- Core modules (`core/`, `economy/`)
- Critical paths (crypto, protocol)
- Edge cases in conditionals

## Writing Tests

### Using Fixtures

Available fixtures in `conftest.py`:

```python
# Crypto
def test_example(crypto):
    """Single crypto identity."""
    assert crypto.node_id is not None

def test_pair(crypto_pair):
    """Sender/receiver pair."""
    sender, receiver = crypto_pair

# Database
@pytest.mark.asyncio
async def test_ledger(ledger):
    """Isolated in-memory ledger."""
    await ledger.record_debt(peer_id, 100)

# Nodes
@pytest.mark.asyncio
async def test_nodes(node_factory):
    """Create test nodes."""
    nodes = await node_factory.create(3, start=True)
    await node_factory.connect_all()

# Mocks
def test_ai(mock_gpu, mock_ollama):
    """Mocked AI operations."""
    pass

def test_chain(mock_chain):
    """Mocked blockchain."""
    mock_chain.fund_account(address, 1000 * 10**18)
```

### Test Structure

```python
"""
Module Description
==================

[UNIT/INTEGRATION/E2E] Brief description.
"""

import pytest
import pytest_asyncio


class TestFeature:
    """Test group description."""
    
    def test_sync_operation(self, fixture):
        """Test sync code."""
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_async_operation(self, ledger):
        """Test async code."""
        result = await ledger.get_balance(peer_id)
        assert result >= 0
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_slow_operation(self, node_factory):
        """Slow test (marked for skipping)."""
        pass
```

### Markers

```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.e2e           # End-to-end test
@pytest.mark.stress        # Stress test
@pytest.mark.slow          # Slow test
@pytest.mark.gpu           # Requires GPU
```

### Async Tests

Always use `@pytest.mark.asyncio`:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_operation()
    assert result is not None
```

### Mocking

```python
from unittest.mock import MagicMock, AsyncMock, patch

def test_with_mock():
    mock_obj = MagicMock()
    mock_obj.method.return_value = "mocked"
    
    # Async mock
    async_mock = AsyncMock(return_value={"result": "ok"})

# Patching
@patch("module.function")
def test_patched(mock_func):
    mock_func.return_value = "patched"
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements/core.txt
        pip install -r requirements/dev.txt
    
    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml --cov-fail-under=80
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
```

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest tests/unit/ -v --tb=short
        language: system
        pass_filenames: false
        always_run: true
```

## Troubleshooting

### Common Issues

**1. Database Locked**

Solution: Use `isolated_db` fixture for per-test databases.

```python
@pytest.mark.asyncio
async def test_db(isolated_db):
    ledger = Ledger(isolated_db)
```

**2. Event Loop Closed**

Solution: Use function-scoped `event_loop` fixture (default).

**3. Import Errors**

Solution: Ensure project root is in PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

**4. Slow Tests Blocking CI**

Solution: Use markers and skip:

```bash
pytest -m "not slow"
```

**5. GPU Tests Failing**

Solution: Use `mock_gpu` fixture or skip:

```python
@pytest.mark.gpu
def test_gpu_feature():
    pytest.importorskip("torch")
```

### Debug Mode

```bash
# Verbose output
pytest -vvv

# Stop on first failure
pytest -x

# Enter debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Show local variables
pytest -l
```

## Stress Testing

### Running Stress Tests

```bash
# Default (100 clients, 60s)
python tests/stress_test.py

# High load
python tests/stress_test.py --clients 500 --duration 120

# Target specific node
python tests/stress_test.py --host 192.168.1.100 --port 8468
```

### Interpreting Results

```
[STRESS] RESULT: [PASS] Error rate below 1%
```

| Metric | Good | Warning | Fail |
|--------|------|---------|------|
| Error Rate | <1% | 1-5% | >5% |
| P99 Latency | <100ms | 100-500ms | >500ms |
| Memory | Stable | Slow growth | Leak |

### Finding Memory Leaks

1. Run stress test with memory tracking
2. Check `peak_memory_mb` growth
3. Use `tracemalloc` for detailed analysis:

```python
import tracemalloc
tracemalloc.start()
# ... run tests ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')[:10]
```

## Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── stress_test.py           # Load testing
├── unit/
│   ├── core/
│   │   ├── test_crypto.py
│   │   ├── test_wire.py
│   │   └── test_merkle.py
│   ├── economy/
│   │   ├── test_ledger.py
│   │   └── test_trust.py
│   └── agents/
│       └── test_manager.py
├── integration/
│   ├── vpn/
│   │   └── test_socks.py
│   ├── dht/
│   │   └── test_kademlia.py
│   └── chain/
│       └── test_settlement.py
└── e2e/
    └── test_full_cycle.py
```

## Best Practices

1. **One assertion per test** (when possible)
2. **Descriptive names**: `test_ledger_blocks_peer_when_debt_exceeds_limit`
3. **Use fixtures** for setup/teardown
4. **Mark slow tests** with `@pytest.mark.slow`
5. **Isolate tests**: No shared state between tests
6. **Clean up**: Use fixtures with cleanup (yield pattern)
7. **Document edge cases**: Comments for non-obvious tests

