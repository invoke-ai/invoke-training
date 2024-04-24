def import_xformers():
    try:
        import xformers  # noqa: F401
    except ImportError:
        raise ImportError(
            "xformers is not installed. Either set `xformers = False` in your training config, or install it using "
            "`pip install xformers>=0.0.23`."
        )
