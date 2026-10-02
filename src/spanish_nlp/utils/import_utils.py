from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version

from packaging.version import Version


class ModuleUpgradeRequiredError(ImportError):
    """El módulo opcional está instalado pero su versión es muy vieja."""


def assert_optional_import(
    module_name: str,
    *,
    package_name: str
    | None = None,  # if PyPi name is not the same (sklearn -> scikit-learn)
    extra: str | None = None,  # library extras: mylib[extra]
    min_version: str | None = None,
    lib_name: str = "spanish_nlp",
) -> None:
    """
    Assert an optional dependency before importing.
    Taken and adapted from polars/_dependencies.py

    Parameters
    ----------
    module_name : str
        Name of the dependency to import.
    err_prefix : str, optional
        Error prefix to use in the raised exception (appears before the module name).
    err_suffix: str, optional
        Error suffix to use in the raised exception (follows the module name).
    min_version : {str, tuple[int]}, optional
        If a minimum module version is required, specify it here.
    min_err_prefix : str, optional
        Override the standard "requires" prefix for the minimum version error message.
    install_message : str, optional
        Override the standard "Please install it using..." exception message fragment.

    Examples
    --------
    >>> from polars._dependencies import import_optional
    >>> import_optional(
    ...     "definitely_a_real_module",
    ...     err_prefix="super-important package",
    ... )  # doctest: +SKIP
    ImportError: super-important package 'definitely_a_real_module' not installed.
    Please install it using the command `pip install definitely_a_real_module`.
    """
    root = module_name.split(".", 1)[0]
    dist = package_name or root

    try:
        import_module(module_name)
    except ImportError:
        install_pip = (
            f"pip install '{lib_name}[{extra}]'" if extra else f"pip install {dist}"
        )
        install_uv = f"uv add '{lib_name}[{extra}]'" if extra else f"uv add {dist}"
        raise ModuleNotFoundError(
            f"'{module_name}' not found.\nPlease install with either: \n- {install_pip}\n- {install_uv}"
        ) from None

    if min_version:
        try:
            found = Version(dist_version(dist))
        except PackageNotFoundError:
            return None  # no podemos verificar; no bloqueamos
        if found < Version(min_version):
            raise ModuleUpgradeRequiredError(
                f"{lib_name} requiere {dist}>={min_version} (encontrado {found})"
            )


if __name__ == "__main__":
    assert_optional_import("polars", lib_name="spanish_nlp")
