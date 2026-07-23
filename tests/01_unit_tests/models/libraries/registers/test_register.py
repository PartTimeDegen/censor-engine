from unittest.mock import MagicMock, call, patch

import pytest

from censor_engine.models.libraries.registries.registry import (
    LIBRARIES_BASE_FOLDER,
    Registry,
)


class TestRegistry:
    def test_init(self):
        registry = Registry("dummy")

        assert registry._library_import_path == (
            f"{LIBRARIES_BASE_FOLDER}.dummy"
        )

    class TestAutoRegister:
        @patch("importlib.import_module")
        @patch("pkgutil.walk_packages")
        def test_baseline(
            self,
            mock_walk_packages,
            mock_import_module,
        ):
            registry = Registry("example")

            package = MagicMock()
            package.__path__ = ["fake_path"]

            mock_import_module.return_value = package

            module1 = MagicMock(name="mod1")
            module1.name = f"{LIBRARIES_BASE_FOLDER}.libraries.example.mod1"
            module1.ispkg = False

            module2 = MagicMock(name="mod2")
            module2.name = f"{LIBRARIES_BASE_FOLDER}.libraries.example.mod2"
            module2.ispkg = False

            package_module = MagicMock(name="subpackage")
            package_module.name = (
                f"{LIBRARIES_BASE_FOLDER}.libraries.example.subpackage"
            )
            package_module.ispkg = True

            mock_walk_packages.return_value = [
                module1,
                package_module,
                module2,
            ]

            registry._auto_register()

            mock_import_module.assert_has_calls(
                [
                    call(f"{LIBRARIES_BASE_FOLDER}.example"),
                    call(f"{LIBRARIES_BASE_FOLDER}.libraries.example.mod1"),
                    call(f"{LIBRARIES_BASE_FOLDER}.libraries.example.mod2"),
                ]
            )

            assert mock_import_module.call_count == 3

    class TestRegister:
        def test_baseline(self):
            registry = Registry("dummy")
            registry.modules_loaded = True

            @registry.register()
            class MyClass:
                pass

            assert registry.get("MyClass") is MyClass
            assert registry.get_all() == {"MyClass": MyClass}

        def test_register_duplicate_class_raises(self):
            registry = Registry("dummy")
            registry.modules_loaded = True

            @registry.register()
            class MyClass:
                pass

            with pytest.raises(
                ValueError,
                match="MyClass is already registered",
            ):
                registry.register()(MyClass)

    class TestGet:
        def test_baseline(self):
            registry = Registry("dummy")
            registry.modules_loaded = True

            @registry.register()
            class MyClass:
                pass

            assert registry.get("MyClass") is MyClass

        def test_get_missing_class_raises(self):
            registry = Registry("dummy")
            registry.modules_loaded = True

            with pytest.raises(
                KeyError,
                match="DoesNotExist is not registered",
            ):
                registry.get("DoesNotExist")

        @patch.object(Registry, "get_all")
        def test_get_calls_get_all_when_not_loaded(
            self,
            mock_get_all,
        ):
            registry = Registry("dummy")

            class MyClass:
                pass

            registry._registry["MyClass"] = MyClass

            def side_effect():
                registry.modules_loaded = True
                return registry._registry.copy()

            mock_get_all.side_effect = side_effect

            result = registry.get("MyClass")

            assert result is MyClass
            mock_get_all.assert_called_once()

    class TestGetAll:
        def test_baseline(self):
            registry = Registry("dummy")
            registry.modules_loaded = True

            @registry.register()
            class MyClass:
                pass

            registry_copy = registry.get_all()

            assert registry_copy == registry._registry
            assert registry_copy is not registry._registry

            registry_copy.clear()

            assert "MyClass" in registry._registry

        @patch.object(Registry, "_auto_register")
        def test_get_all_calls_auto_register_once(
            self,
            mock_auto_register,
        ):
            registry = Registry("dummy")

            registry.get_all()

            mock_auto_register.assert_called_once()
            assert registry.modules_loaded is True

            registry.get_all()

            mock_auto_register.assert_called_once()
