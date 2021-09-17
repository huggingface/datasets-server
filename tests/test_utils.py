from datasets_preview_backend.utils import get_bool_value, get_int_value, get_str_value


def test_get_bool_value():
    assert get_bool_value({"KEY": "True"}, "KEY", False) is True
    assert get_bool_value({"KEY": "true"}, "KEY", False) is True
    assert get_bool_value({"KEY": True}, "KEY", False) is True
    assert get_bool_value({"KEY": "True"}, "DOESNOTEXIST", False) is False
    assert get_bool_value({"KEY": ""}, "KEY", False) is False
    assert get_bool_value({}, "KEY", False) is False
    assert get_bool_value({}, "KEY", True) is True
    # default value type is not constrained
    assert get_bool_value({}, "KEY", None) is None
    assert get_bool_value({}, "KEY", "string") == "string"


def test_get_int_value():
    default = 456
    assert get_int_value({"KEY": "123"}, "KEY", default) == 123
    assert get_int_value({"KEY": 123}, "KEY", default) == 123
    assert get_int_value({"KEY": "123"}, "DOESNOTEXIST", default) == default
    assert get_int_value({"KEY": ""}, "KEY", default) == default
    assert get_int_value({}, "KEY", default) == default
    # default value type is not constrained
    assert get_int_value({}, "KEY", None) is None
    assert get_int_value({}, "KEY", "string") == "string"


def test_get_str_value():
    default = "string"
    assert get_str_value({}, "KEY", default) == default
    # Empty string is ignored
    assert get_str_value({"KEY": ""}, "KEY", default) == default
    assert get_str_value({"KEY": "test"}, "KEY", default) == "test"
    assert get_str_value({"KEY": "None"}, "KEY", default) == "None"
    assert get_str_value({"KEY": "test"}, "DOESNOTEXIST", default) == default
    # default value type is not constrained
    assert get_str_value({}, "KEY", None) is None
    assert get_str_value({}, "KEY", 123) == 123
