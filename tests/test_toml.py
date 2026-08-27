"""Tests for the TOML fragment helpers.

``deeperfly._toml`` exists so that a project's files can be *composed by concatenating
fragments* rather than serialized from a merged dict -- which is what keeps every comment
in the packaged config alive through a `deeperfly project new`. That makes the section
extractors load-bearing: a fragment that is silently truncated produces a project whose
skeleton is missing half its definition, and the truncation is not visible until something
downstream reads a key that is no longer there.
"""

from __future__ import annotations

import tomllib

import numpy as np
import pytest

from deeperfly import _toml


def test_value_nests():
    """``[skeleton].symmetries`` is an array of arrays, so ``value`` has to recurse."""
    assert _toml.value([["a", "b"], ["c", "d"]]) == '[["a", "b"], ["c", "d"]]'
    assert _toml.value([1, 2.5, True]) == "[1, 2.5, true]"
    assert _toml.value(np.array([[0, 19], [1, 20]])) == "[[0, 19], [1, 20]]"
    # A str is a scalar, never iterated into characters.
    assert _toml.value("ab") == '"ab"'


# A multi-line array of arrays: every continuation line begins with `[`, which is exactly
# what a line-local "is this a table header?" test mistakes for the end of the section.
_ARRAY_OF_ARRAYS = """\
[skeleton]
name = "toy"
point_names = ["l_a", "r_a"]
symmetries = [
    ["l_a", "r_a"],
]

[skeleton.limb_palette]
body = "#000000"

[cameras.defaults]
distance = 1.0
"""


def test_extract_section_is_not_ended_by_an_array_continuation_line():
    """The regression: `["l_a", "r_a"],` is a value, not a `[table]` header.

    Before this was depth-aware, extracting ``[skeleton]`` from a config with a multi-line
    array of arrays cut the text off inside the array -- so the fragment did not even parse,
    and the packaged skeleton lost its symmetry pairs the moment a project was seeded.
    """
    section = _toml.extract_section(_ARRAY_OF_ARRAYS, "skeleton")
    parsed = tomllib.loads(section)  # would raise on a truncated array
    assert set(parsed) == {"skeleton"}
    assert parsed["skeleton"]["symmetries"] == [["l_a", "r_a"]]
    # Sub-tables of the section come along; the next top-level table does not.
    assert parsed["skeleton"]["limb_palette"] == {"body": "#000000"}


def test_top_level_tables_ignores_array_continuation_lines():
    assert _toml.top_level_tables(_ARRAY_OF_ARRAYS) == ["skeleton", "cameras"]


def test_extract_tables_ignores_array_continuation_lines():
    kept = tomllib.loads(_toml.extract_tables(_ARRAY_OF_ARRAYS, ["skeleton"]))
    assert set(kept) == {"skeleton"}
    assert kept["skeleton"]["symmetries"] == [["l_a", "r_a"]]


def test_a_bracket_inside_a_comment_or_a_string_is_not_a_header():
    """Depth counting has to skip comments and quoted text, or it desynchronizes.

    A stray unbalanced bracket in prose would leave the scanner at nonzero depth for the
    rest of the file and swallow every later section.
    """
    text = """\
[a]
# a comment mentioning [b] and an unbalanced [
note = "a string with ] in it"

[b]
x = 1
"""
    assert _toml.top_level_tables(text) == ["a", "b"]
    assert set(tomllib.loads(_toml.extract_section(text, "a"))) == {"a"}


def test_the_packaged_config_round_trips_through_every_extractor():
    """The real artifact, not a fixture: a fragment lifted out of it must still parse."""
    from deeperfly.config import DEFAULT_CONFIG_PATH

    text = DEFAULT_CONFIG_PATH.read_text()
    whole = tomllib.loads(text)
    for name in _toml.top_level_tables(text):
        one = tomllib.loads(_toml.extract_tables(text, [name]))
        assert set(one) == {name}, name
    # The packaged config declares NO [skeleton] table -- the skeleton is a file of its
    # own -- so the extractor has nothing to lift, and says so rather than returning an
    # empty fragment that would parse as a skeleton with no points.
    assert "skeleton" not in whole
    with pytest.raises(ValueError, match="no .skeleton. table"):
        _toml.extract_section(text, "skeleton")
    # A lifted [pipeline] section must mean exactly what it meant in the whole file.
    lifted = tomllib.loads(_toml.extract_section(text, "pipeline"))["pipeline"]
    assert lifted == whole["pipeline"]
