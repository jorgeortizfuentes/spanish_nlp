from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version

from packaging.version import Version


class ModuleUpgradeRequiredError(ImportError):
    """El módulo opcional está instalado pero su versión es muy vieja."""


def assert_optional_import(
    module_name: str,
    *,
    package_name: str | None = None,  # if PyPi name differs (sklearn -> scikit-learn)
    extra: str | None = None,  # library extras: mylib[extra]
    min_version: str | None = None,
    lib_name: str = "spanish_nlp",
) -> None:
    """
    Assert that an optional dependency is available before using it.
    If not, print a personalized message.

    The module is imported only to check that it exists; it is not returned,
    so import it normally afterwards. Adapted from polars/_dependencies.py.

    Parameters
    ----------
    module_name : str
        Importable name of the dependency (e.g. ``"sklearn"``). Dotted names
        are allowed (e.g. ``"emoji.unicode_codes"``).
    package_name : str, optional
        Distribution name on PyPI, when it differs from the module name
        (e.g. ``"scikit-learn"`` for ``sklearn``). Defaults to the root of
        ``module_name``. Used for the install hint and the version check.
    extra : str, optional
        Name of the extra of this library that provides the dependency
        (``lib_name[extra]``). If given, the install hint points to the extra;
        otherwise it points to ``package_name`` directly.
    min_version : str, optional
        Minimum required version of the distribution, in PEP 440 format
        (e.g. ``"2.0"``). The version is read from the installed package
        metadata, not from ``module.__version__``. If the metadata cannot be
        found, the check is skipped.
    lib_name : str, default "spanish_nlp"
        Name of this library, used in the install hint and error messages.

    Raises
    ------
    ModuleNotFoundError
        If ``module_name`` cannot be imported. The message includes the
        ``pip`` and ``uv`` commands to install it.
    ModuleUpgradeRequiredError
        If the installed version is lower than ``min_version``.

    Examples
    --------
    >>> assert_optional_import(
    ...     "foobar", extra="emoji", libname="spanish_nlp"
    ... )  # doctest: +SKIP
    ModuleNotFoundError: 'foobar' not found.
    Please install with either:
    - pip install 'spanish_nlp[emoji]'
    - uv add 'spanish_nlp[emoji]'

    >>> assert_optional_import(
    ...     "sklearn", package_name="scikit-learn", min_version="1.3"
    ... )  # doctest: +SKIP
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
