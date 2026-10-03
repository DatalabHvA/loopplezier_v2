import streamlit as st
import pandas as pd
import geopandas as gpd
from streamlit_folium import st_folium
import folium
from functions_verplichte_punten import *
import numpy as np
from map_utils import base_map, DEFAULT_WEIGHTS, weights_key, weight_form

st.set_page_config(layout='wide')


@st.cache_data(show_spinner="Route berekenen...")
def _cached_mandatory_route(_gdf, start, end, mandatory, max_dist, wkey):
    """Cache de GRASP-route per combinatie van start/eind/tussenpunten/afstand/gewichten.

    GRASP is gerandomiseerd; door te cachen krijg je bij dezelfde invoer ook
    dezelfde route terug in plaats van elke rerun een andere.
    """
    return calculate_mandatory_route(
        gdf=_gdf,
        start=start,
        end=end,
        mandatory_nodes=list(mandatory),
        max_dist=max_dist,
    )


@st.cache_data
def load_data():
    gdf = gpd.read_feather('./data/gdf.feather')
    gdf = gdf.reset_index(drop=True)
    gdf = calculate_new_column(gdf, **DEFAULT_WEIGHTS)

    nodes = gpd.read_feather('./data/nodes.feather').to_crs('EPSG:4326')
    nodes = nodes.reset_index().rename(columns={'osmid': 'knooppunt'})
    return (gdf, nodes)


def calculate_new_column(gdf, ovl, bomen, water, monumenten, wegen, parken, toiletten, verkeerslichten, wegdekkwaliteit, horeca, kerk, winkels, groen, kampioen, waarnemingen, ov, schaduw, colum_name='Score'):
    # Score op basis van gewichten ingevuld op streamlit
    gdf['score_totaal'] = (
        gdf['score_bomen'] * bomen +
        gdf['score_ovl'] * ovl +
        gdf['score_water'] * water +
        gdf['score_monumenten'] * monumenten +
        gdf['score_wegen'] * wegen +
        gdf['score_park'] * parken +
        gdf['score_verkeerslichten'] * verkeerslichten +
        gdf['score_horeca'] * horeca +
        gdf['score_OV'] * ov +
        gdf['score_schaduw'] * schaduw +
        gdf['score_winkels'] * winkels)
    return gdf


def bezoekvolgorde(_df_route, mandatory_nodes):
    """In welke volgorde doet de route de verplichte punten aan?"""
    if not mandatory_nodes or _df_route is None or len(_df_route) == 0:
        return []

    route_nodes = [_df_route.iloc[0]["u"]]
    for _, row in _df_route.iterrows():
        route_nodes.append(row["v"])

    volgorde = []
    seen = set()
    for node in route_nodes:
        if node in mandatory_nodes and node not in seen:
            volgorde.append(node)
            seen.add(node)
    return volgorde


def route_layer(_df_route, _nodes, volgorde):
    """FeatureGroup met de route plus genummerde markers voor de verplichte
    punten; wordt los van de (statische) basiskaart bijgewerkt via
    st_folium(feature_group_to_add=...)."""
    fg = folium.FeatureGroup(name='Route')

    for geom in _df_route.geometry:
        if geom is None:
            continue
        parts = geom.geoms if geom.geom_type == 'MultiLineString' else [geom]
        for part in parts:
            folium.PolyLine([(y, x) for x, y in part.coords], color='#3388ff', weight=5).add_to(fg)

    order_dict = {node: i + 1 for i, node in enumerate(volgorde)}
    punten = _nodes[_nodes["knooppunt"].isin(order_dict)]

    for _, row in punten.iterrows():
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            icon=folium.DivIcon(html=f"""
                <div style="
                    background-color:red;
                    border-radius:50%;
                    width:24px;
                    height:24px;
                    line-height:24px;
                    text-align:center;
                    color:white;
                    font-weight:bold;
                    font-size:12px;
                    border:2px solid white;
                ">{order_dict[row['knooppunt']]}</div>
            """)
        ).add_to(fg)

    return fg


def main():
    st.title("Verplichte tussenpunten")
    st.caption("Hogeschool van Amsterdam — Bereken een route die onderweg langs punten komt die jij kiest")

    with st.expander("Over deze kaart — wat zie ik hier?"):
        st.markdown("""
Deze kaart berekent een route die **onderweg langs punten komt die jij zelf opgeeft** —
bijvoorbeeld een winkel, een bankje of een plek waar je iemand ophaalt.

**Jouw eigen score**
In het menu links geef je elke omgevingsfactor een gewicht tussen –10 en +10. Klik op
**Bereken** om de kaart te kleuren:
🟢 groen = aantrekkelijk &nbsp;·&nbsp; 🟡 geel = neutraal &nbsp;·&nbsp; 🔴 rood = minder aantrekkelijk

**Een route berekenen**
Vul een start- en eindknooppunt in (de zwarte punten op de kaart; beweeg je muis over een
punt om het nummer te zien). Zet in het veld *Verplichte tussenpunten* de nummers van de
punten waar je langs wilt, gescheiden door komma's, en klik op **Route toevoegen**.

**De volgorde**
Je hoeft de volgorde niet zelf te bepalen: het algoritme kiest de handigste
volgorde binnen de maximale afstand. De rode genummerde bolletjes op de kaart laten zien
in welke volgorde je de punten aandoet; rechts staat dezelfde volgorde als lijst.
        """)

    (gdf, nodes) = load_data()

    weights, calculate_button = weight_form()

    st.sidebar.divider()
    st.sidebar.subheader("Route berekenen")

    with st.sidebar.form("Route"):
        c1, c2 = st.columns(2)
        start = c1.number_input("Start", 0, 3100, 2913, 1, key="start")
        end = c2.number_input("Eind", 0, 3100, 3045, 1, key="end")
        max_dist = st.number_input("Max. afstand (m)", 500, 20000, 9000, 100, key="max_dist")
        mandatory_text = st.text_input("Verplichte tussenpunten (gescheiden door komma's)", "909, 715")
        add_route = st.form_submit_button("Route toevoegen", use_container_width=True)

    mandatory_nodes = []
    if mandatory_text.strip():
        try:
            mandatory_nodes = [int(x.strip()) for x in mandatory_text.split(",") if x.strip()]
        except ValueError:
            st.sidebar.error("Gebruik alleen knooppuntnummers, gescheiden door komma's.")

    ss = st.session_state
    STATE_KEYS = ("tp_route", "tp_distance", "tp_score", "tp_volgorde", "tp_weights")

    # 'Bereken' = alleen de kaart herkleuren; verwijder een eventuele actieve route
    if calculate_button:
        for k in STATE_KEYS:
            ss.pop(k, None)

    # 'Route toevoegen' = nieuwe route berekenen en bewaren in session_state
    if add_route:
        gdf = calculate_new_column(gdf, **weights)

        df_route, distance, score = _cached_mandatory_route(
            gdf, start=start, end=end, mandatory=tuple(mandatory_nodes),
            max_dist=max_dist, wkey=weights_key(weights),
        )

        if df_route is None or len(df_route) == 0:
            st.error("Geen route gevonden die alle verplichte punten bezoekt")
            for k in STATE_KEYS:
                ss.pop(k, None)
        else:
            df_route = gpd.GeoDataFrame(df_route, geometry="geometry", crs=gdf.crs)
            ss["tp_route"] = df_route
            ss["tp_distance"] = distance
            ss["tp_score"] = score
            ss["tp_volgorde"] = bezoekvolgorde(df_route, mandatory_nodes)
            ss["tp_weights"] = weights

    # ---- Render op basis van de route in session_state ----
    df_route = ss.get("tp_route")
    route = df_route is not None
    distance = ss.get("tp_distance", 0)
    score = ss.get("tp_score", 0)
    volgorde = ss.get("tp_volgorde", [])

    if route:
        current_weights = ss["tp_weights"]
    elif calculate_button:
        current_weights = weights
    else:
        current_weights = DEFAULT_WEIGHTS

    gdf = calculate_new_column(gdf, **current_weights)

    col_map, col_side = st.columns([3, 2], gap="medium")

    with col_map:
        # Lege FeatureGroup i.p.v. None: st_folium haalt een eerder toegevoegde
        # laag niet weg als je None doorgeeft, waardoor een gewiste route
        # zichtbaar bleef.
        st_folium(
            base_map(gdf, nodes, current_weights),
            feature_group_to_add=route_layer(df_route, nodes, volgorde) if route else folium.FeatureGroup(name="Route"),
            width=700, height=620, returned_objects=[], key="tussenpunten_map",
        )

    with col_side:
        if route:
            m1, m2 = st.columns(2)
            m1.metric("Afstand", f"{distance / 1000:.2f} km")
            m2.metric("Score", f"{score:.2f}")
            if score == -10:
                st.warning("Niet mogelijk om alle waypoints te bezoeken")

            if volgorde:
                st.markdown("**Volgorde van de tussenpunten**")
                st.markdown("\n".join(f"{i}. knooppunt **{n}**" for i, n in enumerate(volgorde, 1)))
                st.caption("Dezelfde nummers staan als rode bolletjes op de kaart.")
        else:
            st.markdown("### Zo begin je")
            st.markdown("""
**1.** Stel links de gewichten in en klik **Bereken** — de kaart kleurt mee.

**2.** Zoek op de kaart je start- en eindknooppunt (de zwarte punten).

**3.** Vul beide nummers links in, zet de knooppunten waar je langs wilt in het
veld *Verplichte tussenpunten*, en klik **Route toevoegen**.
            """)
            st.info(
                "De route komt langs alle punten die je opgeeft. Het algoritme kiest zelf "
                "de handigste volgorde; die zie je daarna hier terug.",
                icon="📍",
            )


# Run the app
if __name__ == '__main__':
    main()
