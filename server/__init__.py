# Re-export app for backward compatibility (tests use `from server import app`)
from server.app import app  # noqa: F401
