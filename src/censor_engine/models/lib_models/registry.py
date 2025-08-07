import importlib
import pkgutil
import re
from collections.abc import Callable
from pathlib import Path


class Registry:
    def __init__(self, package: str):
        self.package = package
        self._registry: dict[str, type] = {}
        self._loaded = False

    def camel_to_snake(self, name: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    def register(self) -> Callable:
        """Decorator to register a class."""

        def decorator(cls):
            if not self._loaded:
                self._auto_register()
            key = self.camel_to_snake(cls.__name__)
            self._registry[key] = cls
            return cls

        return decorator

    def _auto_register(self):
        module = importlib.import_module(self.package)
        package_dir = Path(module.__file__).parent  # type: ignore
        for _, module_name, is_package in pkgutil.iter_modules(
            [str(package_dir)]
        ):
            if not is_package:
                importlib.import_module(f"{self.package}.{module_name}")
        self._loaded = True

    def get_all(self) -> dict[str, type]:
        """Return all registered classes, loading them if needed."""
        if not self._loaded:
            self._auto_register()
        return dict(self._registry)  # Return a copy for safety
