def dag_option(context, key, default=None):
    """Read manual trigger config first, then DAG params, normalizing blanks."""
    dag_run = context.get("dag_run")
    if dag_run and dag_run.conf and key in dag_run.conf:
        value = dag_run.conf.get(key)
    else:
        value = (context.get("params") or {}).get(key, default)

    if value == "":
        return default
    return value


def dag_option_bool(context, key, default=False):
    value = dag_option(context, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def dag_option_int(context, key, default):
    value = dag_option(context, key, default)
    if value is None:
        return default
    return int(value)


def dag_option_float(context, key, default):
    value = dag_option(context, key, default)
    if value is None:
        return default
    return float(value)
