import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from ._constants import (
    LIBRARIES_BASE_FOLDER,
)

T = TypeVar("T", bound=type)


@dataclass(slots=True)
class Registry:
    library_name: str

    base_folder_prefix: str = LIBRARIES_BASE_FOLDER
    modules_loaded: bool = False

    _registry: dict[str, type] = field(default_factory=dict)

    @property
    def _library_import_path(self) -> str:
        return f"{self.base_folder_prefix}.{self.library_name}"

    # Registering Method
    def _auto_register(self) -> None:
        """
        Import every module inside the package.
        Importing modules triggers registration decorators.

        """
        # Get Base Library
        package = importlib.import_module(self._library_import_path)

        # Iterate Through Packages
        for module in pkgutil.walk_packages(
            package.__path__,
            prefix=f"{self._library_import_path}.",
        ):
            if not module.ispkg:
                importlib.import_module(module.name)

    # Decorator
    def register(self) -> Callable[[T], T]:
        """
        This is a decorator to load the class into the register.

        Returns:
            Class to be loaded into the register

        """

        def decorator(cls: T) -> T:
            name = cls.__name__

            if name in self._registry:
                msg = f"{name} is already registered"
                raise ValueError(msg)

            self._registry[name] = cls
            return cls

        return decorator

    def get_all(self) -> dict[str, type]:
        """
        This method gets all of the registered classes found in the registry.

        Returns:
            Dictionary of the class names and their object

        """
        if not self.modules_loaded:
            self._auto_register()
            self.modules_loaded = True

        return self._registry.copy()

    def get(self, name: str) -> type:
        """
        This method gets a registered classes found in the registry.

        Returns:
            Dictionary of the class names and their object

        """
        if not self.modules_loaded:
            self.get_all()

        try:
            return self._registry[name]
        except KeyError:
            msg = f"{name} is not registered"
            raise KeyError(msg) from None
