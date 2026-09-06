import importlib.util
import inspect
from pathlib import Path

from app.parsers.dahua import DahuaParser
from app.parsers.hikvision import HikvisionParser
from app.parsers.vendor_base import VendorParser


class VendorParserRegistry:

    def __init__(self):
        self._parsers: list[VendorParser] = []

    def register(
        self,
        parser: VendorParser,
    ):
        if not isinstance(parser, VendorParser):
            raise TypeError(
                "Parser must inherit from VendorParser."
            )

        self._parsers.append(parser)

    def find_parser(
        self,
        image_path: Path,
        filesystem_analysis: dict,
    ) -> VendorParser | None:

        for parser in self._parsers:
            if parser.can_parse(
                image_path,
                filesystem_analysis,
            ):
                return parser

        return None

    def discover_plugins(
        self,
        plugins_directory: Path,
    ) -> dict:
        loaded_plugins = []
        plugin_errors = []

        if not plugins_directory.exists():
            return {
                "loaded_plugins": loaded_plugins,
                "plugin_errors": plugin_errors,
            }

        for plugin_path in sorted(
            plugins_directory.glob("*.py")
        ):
            if plugin_path.name.startswith("_"):
                continue

            module_name = (
                f"forenvault_plugin_"
                f"{plugin_path.stem}"
            )

            try:
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    plugin_path,
                )

                if spec is None or spec.loader is None:
                    plugin_errors.append(
                        {
                            "plugin": plugin_path.name,
                            "error": (
                                "Could not create module "
                                "specification."
                            ),
                        }
                    )
                    continue

                module = importlib.util.module_from_spec(
                    spec
                )

                spec.loader.exec_module(module)

                found_parser = False

                for _, obj in inspect.getmembers(
                    module,
                    inspect.isclass,
                ):
                    if (
                        obj is VendorParser
                        or not issubclass(
                            obj,
                            VendorParser,
                        )
                    ):
                        continue

                    if inspect.isabstract(obj):
                        continue

                    parser = obj()

                    self.register(parser)

                    loaded_plugins.append(
                        parser.vendor_name
                    )

                    found_parser = True

                if not found_parser:
                    plugin_errors.append(
                        {
                            "plugin": plugin_path.name,
                            "error": (
                                "No concrete VendorParser "
                                "implementation found."
                            ),
                        }
                    )

            except Exception as error:
                plugin_errors.append(
                    {
                        "plugin": plugin_path.name,
                        "error": str(error),
                    }
                )

        return {
            "loaded_plugins": loaded_plugins,
            "plugin_errors": plugin_errors,
        }


def create_default_registry() -> VendorParserRegistry:

    registry = VendorParserRegistry()

    registry.register(
        HikvisionParser()
    )

    registry.register(
        DahuaParser()
    )

    return registry