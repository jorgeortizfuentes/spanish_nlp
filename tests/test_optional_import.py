import re

import pytest

from spanish_nlp.utils.import_utils import assert_optional_import


def test_assert_optional_import():
    msg_error = (
        "'polars' not found.\n"
        "Please install with either: \n"
        "- pip install 'spanish_nlp[polars]'\n"
        "- uv add 'spanish_nlp[polars]'"
    )
    with pytest.raises(ModuleNotFoundError, match=re.escape(msg_error)):
        assert_optional_import("polars", extra="polars", lib_name="spanish_nlp")
