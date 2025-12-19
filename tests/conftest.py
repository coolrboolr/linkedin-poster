import pytest
import random

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy optional
    np = None


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
def _seed():
    random.seed(1337)
    if np is not None:
        np.random.seed(1337)
