def get_significance(p_value):
    if p_value <= 0.001:
        return "$p\\leq$0.001 (***)"
    elif p_value <= 0.01:
        return "$p\\leq$0.01 (**)"
    elif p_value <= 0.05:
        return "$p\\leq$0.05 (*)"
    else:
        return f"$p=${p_value:.3f}"
