import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def outputs_dir():
    """Ensure the outputs/ directory used by augmentation tests exists."""
    os.makedirs("outputs", exist_ok=True)
