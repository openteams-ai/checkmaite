try:
    # smoke-test for whether UI dependencies installed or not
    import panel  # noqa: F401, ICN001
except ImportError as e:
    raise ImportError("UI components require the 'ui' extra. " "Install with: pip install jatic-ri[ui]") from e
