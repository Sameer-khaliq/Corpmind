import pytest

from corpmind.config import Settings, check_env_file_whitespace, settings


def test_module_level_settings_singleton_resolves():
    """The plan's literal Day 1 checkpoint: `from corpmind.config import
    settings` must work and print a resolved config. tests/conftest.py sets
    placeholder env vars before this module (or any test module) imports
    corpmind.config, since the import itself triggers construction."""
    dumped = settings.model_dump()
    assert dumped["extraction_model"] == "llama-3.1-8b-instant"
    assert dumped["faithfulness_threshold"] == 0.85


def test_missing_google_api_key_fails_loud():
    with pytest.raises(Exception, match="GOOGLE_API_KEY"):
        Settings(GROQ_API_KEY="x", GOOGLE_API_KEY="", TAVILY_API_KEY="x")


def test_valid_settings_construct_cleanly():
    s = Settings(GROQ_API_KEY="x", GOOGLE_API_KEY="x", TAVILY_API_KEY="x")
    assert s.faithfulness_threshold == 0.85
    assert s.extraction_model == "llama-3.1-8b-instant"
    assert s.ENVIRONMENT == "development"


def test_match_confidence_ordering_enforced():
    with pytest.raises(ValueError, match="match_confidence_low"):
        Settings(
            GROQ_API_KEY="x",
            GOOGLE_API_KEY="x",
            TAVILY_API_KEY="x",
            match_confidence_high=0.4,
            match_confidence_low=0.6,
        )


def test_quoted_whitespace_in_api_key_rejected():
    with pytest.raises(ValueError, match="whitespace"):
        Settings(GROQ_API_KEY="x", GOOGLE_API_KEY="x ", TAVILY_API_KEY="x")


def test_env_whitespace_detected(tmp_path):
    bad_env = tmp_path / ".env"
    bad_env.write_text("GROQ_API_KEY=abc123 \n")  # trailing space — the bug this catches
    with pytest.raises(ValueError, match="trailing whitespace"):
        check_env_file_whitespace(bad_env)


def test_env_without_whitespace_passes(tmp_path):
    good_env = tmp_path / ".env"
    good_env.write_text("GROQ_API_KEY=abc123\n")
    check_env_file_whitespace(good_env)  # should not raise


def test_missing_env_file_is_not_an_error(tmp_path):
    check_env_file_whitespace(tmp_path / "does_not_exist.env")
