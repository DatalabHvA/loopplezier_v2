"""Start, eind en tussenpunten kiezen door op de kaart te tikken.

Je hoeft geen knooppunt precies te raken: een tik ergens op de kaart wordt
vastgezet op het dichtstbijzijnde knooppunt. Zo werkt het met een muis, maar ook
met een vinger op een groot smartboard. De keuze staat in ``st.session_state``
en blijft dus bewaard als je naar een andere pagina gaat.
"""
from functools import partial
from math import cos, radians

import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

START, END, VIA = "Start", "Eind", "Tussenpunt"
MODE_KEY = "pick_mode"
STATE = {START: "pick_start", END: "pick_end", VIA: "pick_via"}
ICONS = {START: "🟢", END: "🔴", VIA: "🔵"}
HINTS = {
    START: "Tik op de kaart om het **startpunt** te kiezen.",
    END: "Tik op de kaart om het **eindpunt** te kiezen.",
    VIA: "Tik op de kaart om een **tussenpunt** toe te voegen. Tik er nog eens op om het weg te halen.",
}
MARKER_COLORS = {START: "#1a9641", END: "#d7191c", VIA: "#2b6cb0"}

# Verder dan dit van elk knooppunt = naast het netwerk getikt.
MAX_SNAP_M = 250
METERS_PER_DEGREE = 111_320

TOUCH_CSS = """
<style>
/* Meegroeien met het scherm: Streamlit rekent in rem, dus op een 4K-smartboard
   wordt alles vanzelf groter. Tot Full HD blijft het de normale 16px. */
html { font-size: clamp(16px, 0.75vw, 30px) !important; touch-action: manipulation; }

/* Minder witruimte boven de titel, zodat de kaart de schermhoogte kan vullen. */
[data-testid="stMainBlockContainer"] { padding-top: 3.5rem; }
iframe[title="streamlit_folium.st_folium"] {
    height: max(620px, calc(100vh - 26rem)) !important;
}

/* De zijbalk heeft een vaste breedte in px; laat hem meegroeien met de tekst. */
[data-testid="stSidebar"] { width: 19rem !important; min-width: 19rem !important; }

/* Vingervriendelijke knoppen, schuiven en uitklapmenu's. */
.stButton button, .stFormSubmitButton button,
button[data-variant="segmented_control"] {
    min-height: 3.25rem;
    font-size: 1.1rem;
}
button[data-variant="segmented_control"] { max-width: none; }
[data-testid="stExpander"] summary { min-height: 3.25rem; align-items: center; }
[data-testid="stSlider"] label { margin-bottom: 0.75rem; }
[data-testid="stSlider"] div[style*="translate(-50%, -50%)"] {
    width: 2rem;
    height: 2rem;
}
</style>
"""


# Plotly bepaalt bij een klik welk punt geraakt is aan de hand van de laatste
# hover. Een vinger hovert niet, dus een tik selecteerde het vorige punt. Daarom
# sturen we bij elke aanraking eerst een muisbeweging naar de grafiek.
TOUCH_PLOT_JS = """
<script>
if (!window.loopplezierTouchPlot) {
    window.loopplezierTouchPlot = true;
    document.addEventListener('touchstart', function (event) {
        var plot = event.target.closest && event.target.closest('.js-plotly-plot');
        var surface = plot && plot.querySelector('.nsewdrag');
        if (!surface) return;
        var touch = event.touches[0];
        surface.dispatchEvent(new MouseEvent('mousemove', {
            bubbles: true, clientX: touch.clientX, clientY: touch.clientY,
        }));
    }, true);
}
</script>
"""


def touch_style():
    """Maak de pagina bedienbaar met een vinger, ook op een groot scherm."""
    st.markdown(TOUCH_CSS, unsafe_allow_html=True)
    st.html(TOUCH_PLOT_JS, unsafe_allow_javascript=True)


def nearest_node(nodes, lat, lng):
    """Geef ``(knooppunt, afstand in meters)`` van het knooppunt het dichtst bij een tik."""
    dy = (nodes.geometry.y.to_numpy() - lat) * METERS_PER_DEGREE
    dx = (nodes.geometry.x.to_numpy() - lng) * METERS_PER_DEGREE * cos(radians(lat))
    distances = np.hypot(dx, dy)
    i = int(distances.argmin())
    return int(nodes["knooppunt"].iloc[i]), float(distances[i])


def picked_points():
    """Geef ``(start, eind, tussenpunten)``; niet-gekozen punten zijn ``None`` of leeg."""
    ss = st.session_state
    return ss.get(STATE[START]), ss.get(STATE[END]), list(ss.get(STATE[VIA], []))


def _clear_points(clear_keys):
    for key in (*STATE.values(), *clear_keys):
        st.session_state.pop(key, None)
    st.session_state[MODE_KEY] = START


def _on_tap(map_key, nodes, with_via):
    """Verwerk een tik: zet het punt vast en ga door naar het volgende punt."""
    ss = st.session_state
    tap = (ss.get(map_key) or {}).get("last_clicked")
    if not tap:
        return

    node, distance = nearest_node(nodes, tap["lat"], tap["lng"])
    if distance > MAX_SNAP_M:
        ss["pick_notice"] = "Dat is te ver van een straat. Tik dichter bij een gekleurde lijn."
        return

    mode = ss.get(MODE_KEY) or START
    other = {START: ss.get(STATE[END]), END: ss.get(STATE[START])}.get(mode)
    if node == other and not with_via:
        # Rondjes kunnen alleen via tussenpunten: de Pareto-zoeker vindt er geen.
        ss["pick_notice"] = "Start en eind moeten verschillende punten zijn."
        return

    if mode == VIA:
        via = list(ss.get(STATE[VIA], []))
        ss[STATE[VIA]] = [n for n in via if n != node] if node in via else via + [node]
    else:
        ss[STATE[mode]] = node
        if mode == START:
            ss[MODE_KEY] = END
        elif with_via:
            ss[MODE_KEY] = VIA


def route_bar(with_via=False, clear_keys=()):
    """Balk boven de kaart: wat kies je nu, en de knop om de route te berekenen.

    Geeft ``True`` terug in de rerun waarin op 'Route toevoegen' is getikt.
    'Wissen' haalt de gekozen punten en de ``clear_keys`` (de route) weg.
    """
    ss = st.session_state
    start, end, via = picked_points()
    options = [START, END] + ([VIA] if with_via else [])

    if ss.get(MODE_KEY) not in options:
        ss[MODE_KEY] = START if start is None else END if end is None else options[-1]

    def label(option):
        if option == VIA:
            return f"{ICONS[VIA]} {VIA} ({len(via)})"
        chosen = (start if option == START else end) is not None
        return f"{ICONS[option]} {option}{' ✓' if chosen else ''}"

    c_mode, c_add, c_clear = st.columns([5, 3, 1.5], vertical_alignment="bottom")
    c_mode.segmented_control(
        "Wat kies je op de kaart?", options, key=MODE_KEY, required=True,
        format_func=label, width="stretch",
    )
    add_route = c_add.button(
        "Route toevoegen", type="primary", width="stretch",
        disabled=start is None or end is None or (with_via and not via),
    )
    c_clear.button("Wissen", width="stretch", on_click=_clear_points, args=(clear_keys,))

    notice = ss.pop("pick_notice", None)
    if notice:
        st.warning(notice)
    elif ss[MODE_KEY] != VIA and start is not None and end is not None and not with_via:
        st.caption("Tik op **Route toevoegen**, of tik op de kaart om het gekozen punt te verplaatsen.")
    else:
        st.caption(HINTS[ss[MODE_KEY]])
    return add_route


def _marker(row, kind, text):
    # vw is hier de breedte van de kaart: op een groot scherm groeit de marker mee.
    size = "max(34px, 3.5vw)"
    return folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        # niet-interactief: een tik op een marker valt door naar de kaart.
        interactive=False,
        icon=folium.DivIcon(
            icon_size=(0, 0),
            html=f"""
                <div style="
                    background-color:{MARKER_COLORS[kind]};
                    border-radius:50%;
                    width:{size};
                    height:{size};
                    transform:translate(-50%, -50%);
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:white;
                    font-weight:bold;
                    font-size:max(16px, 1.6vw);
                    border:3px solid white;
                    box-shadow:0 1px 4px rgba(0,0,0,.5);
                ">{text}</div>
            """,
        ),
    )


def _add_picked_markers(layer, nodes):
    start, end, via = picked_points()
    by_id = nodes.set_index("knooppunt")
    for node in via:
        _marker(by_id.loc[node], VIA, "+").add_to(layer)
    if start is not None:
        _marker(by_id.loc[start], START, "S").add_to(layer)
    if end is not None:
        _marker(by_id.loc[end], END, "E").add_to(layer)


def tap_map(base, layer, nodes, key, with_via=False):
    """Toon de kaart met ``layer`` plus de gekozen punten, en luister naar tikken."""
    _add_picked_markers(layer, nodes)
    st_folium(
        base,
        feature_group_to_add=layer,
        use_container_width=True,
        height=620,  # ondergrens; TOUCH_CSS rekt de kaart op tot de schermhoogte
        returned_objects=["last_clicked"],
        key=key,
        on_change=partial(_on_tap, key, nodes, with_via),
    )
