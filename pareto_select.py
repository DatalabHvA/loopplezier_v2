"""Helpers voor klikbare Pareto-grafieken (Streamlit native plotly selection).

De Pareto-grafieken tonen als eerste trace (curve 0) alle mogelijke routes en
optioneel als tweede trace (curve 1) de geselecteerde route (rode stip). Een
klik op een punt levert via ``st.plotly_chart(on_select="rerun")`` een event op;
``clicked_point_index`` vertaalt dat naar de positie (rij-index) van het
aangeklikte punt in het Pareto-dataframe.
"""


def clicked_point_index(event, n_points):
    """Geef de positie (0..n_points-1) van het aangeklikte Pareto-punt, of None.

    Alleen punten op de hoofd-trace (curve 0 = "alle routes") tellen mee; een
    klik op de rode 'geselecteerde route'-stip (curve 1) wordt genegeerd.
    """
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return None

    for p in points:
        if int(p.get("curve_number", 0)) != 0:
            continue
        idx = p.get("point_index")
        if idx is None:
            idx = p.get("point_number")
        if idx is not None and 0 <= int(idx) < n_points:
            return int(idx)

    return None
