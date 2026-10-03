import streamlit as st
import pandas as pd
import geopandas as gpd
from streamlit_folium import st_folium
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from functions import *
import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import linemerge
from pareto_select import clicked_point_index
from map_utils import base_map, DEFAULT_WEIGHTS, weights_key, weight_form
st.set_page_config(layout='wide')


@st.cache_data(show_spinner="Routes berekenen...")
def _cached_routes(_gdf, start, end, L_min, L_max, wkey):
    """Cache route-berekening per combinatie van start/eind/afstand/gewichten."""
    return generate_routes(_gdf, start=start, end=end, L_min=L_min, L_max=L_max,
                           max_lens=[5, 10, 25, 50, 100, 250])


@st.cache_data(show_spinner=False)
def _bankjes_voor_route(route_tuple, _gdf):
    """Welke bankjes liggen binnen 50m van deze route? Gecachet per routepad."""
    df = pareto_path_to_gdf(_gdf, list(route_tuple))
    route_gdf = df[df.geometry.notna()].copy()
    if route_gdf.crs is None:
        route_gdf = route_gdf.set_crs("EPSG:4326")
    route_proj = route_gdf.to_crs("EPSG:3857")
    route_buffer = route_proj.geometry.unary_union.buffer(50)
    bio_proj = load_bankjes_proj()
    return bio_proj[bio_proj.intersects(route_buffer)].to_crs("EPSG:4326")


@st.cache_data
def load_data():
    gdf = gpd.read_feather('./data/gdf.feather')
    gdf = gdf.reset_index(drop=True)
    gdf = calculate_new_column(gdf, **DEFAULT_WEIGHTS)

    nodes = gpd.read_feather('./data/nodes.feather').to_crs('EPSG:4326')
    nodes = nodes.reset_index().rename(columns={'osmid': 'knooppunt'})
    return (gdf, nodes)


@st.cache_data
def load_bankjes_proj():
    # Bankjes één keer inlezen + projecteren (i.p.v. bij elke rerun).
    bio = gpd.read_feather('data/bankjes_clean.feather')
    if bio.crs is None:
        bio = bio.set_crs("EPSG:4326")
    return bio.to_crs("EPSG:3857")


def style_function(feature):
    cmap = cm.RdYlGn  # Choose a continuous colormap (11 colors)
    value = feature['properties']['score_totaal']  # Get the value from your column
    normalized_value = (0.5 * value) + 0.5
    color = mcolors.rgb2hex(cmap(normalized_value))  # Map the value to a color
    return {'color': color}


def style_function_route(feature):
    return {'weight': 5}


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


def route_layer(_df_route, bankjes_gdf):
    """FeatureGroup met de route + bijbehorende bankjes; wordt los van de
    (statische) basiskaart bijgewerkt via st_folium(feature_group_to_add=...).
    bankjes_gdf is pre-berekend via _bankjes_voor_route (gecachet)."""
    fg = folium.FeatureGroup(name='Route')
    for geom in _df_route.geometry:
        if geom is None:
            continue
        parts = geom.geoms if geom.geom_type == 'MultiLineString' else [geom]
        for part in parts:
            folium.PolyLine([(y, x) for x, y in part.coords], color='#3388ff', weight=5).add_to(fg)
    for point in bankjes_gdf.geometry:
        folium.Marker(
            [point.y, point.x],
            icon=folium.features.CustomIcon('./bankje.png', icon_size=(30, 30))
        ).add_to(fg)
    return fg


def main():
    # Title and description

    st.title("Route met bankjes")
    st.caption("Hogeschool van Amsterdam — Vind een route waarop je onderweg regelmatig kunt zitten")

    with st.expander("Over deze kaart — wat zie ik hier?"):
        st.markdown("""
Deze kaart zoekt wandelroutes waarop je **onderweg regelmatig een bankje** tegenkomt —
handig als je liever niet lang achter elkaar loopt.

**Jouw eigen score**
In het menu links geef je elke omgevingsfactor een gewicht tussen –10 en +10. Klik op
**Bereken** om de kaart te kleuren:
🟢 groen = aantrekkelijk &nbsp;·&nbsp; 🟡 geel = neutraal &nbsp;·&nbsp; 🔴 rood = minder aantrekkelijk

**Een route berekenen**
Vul een start- en eindknooppunt in (de zwarte punten op de kaart; beweeg je muis over een
punt om het nummer te zien) en klik op **Route toevoegen**. De gevonden bankjes langs de route
worden als icoon op de kaart gezet.

**Het langste stuk zonder bankje, en het Pareto-front**
Het *langste stuk* is de grootste afstand die je op de route aflegt zonder een bankje
tegen te komen. Een route met veel bankjes
heeft vaak een lagere omgevingsscore — er is dus geen enkele beste route. Het algoritme
zoekt alle routes die je niet kunt verbeteren zonder op de andere as in te leveren: het
*Pareto-front*. De grafiek rechts toont die alternatieven; klik op een punt om die route
op de kaart te zien.
        """)

    (gdf, nodes) = load_data()

    weights, calculate_button = weight_form()

    st.sidebar.divider()
    st.sidebar.subheader("Route berekenen")

    with st.sidebar.form("Route"):
        c1, c2 = st.columns(2)
        start = c1.number_input("Start", 0, 3100, 2913, 1, key="start")
        end = c2.number_input("Eind", 0, 3100, 3045, 1, key="end")
        c3, c4 = st.columns(2)
        min_dist = c3.number_input("Min. afstand (m)", 500, 10000, 500, 100, key="min_dist")
        max_dist = c4.number_input("Max. afstand (m)", 500, 10000, 3000, 100, key="max_dist")
        max_bankjes_afstand = st.number_input("Max. afstand tussen bankjes (m)", 100, 2000, 500, 50)
        add_route = st.form_submit_button("Route toevoegen", use_container_width=True)

    ss = st.session_state
    STATE_KEYS = ("bankjes_df", "bankjes_weights", "bankjes_pos", "bankjes_cid")

    # 'Bereken' = alleen de kaart herkleuren; verwijder een eventuele actieve route
    if calculate_button:
        gdf = calculate_new_column(gdf, **weights)
        for k in STATE_KEYS:
            ss.pop(k, None)

    # 'Route toevoegen' = nieuwe Pareto-set berekenen en bewaren in session_state
    if add_route:
        gdf = calculate_new_column(gdf, **weights)

        pareto = _cached_routes(
            gdf, start=start, end=end, L_min=min_dist, L_max=max_dist,
            wkey=weights_key(weights)
        )

        pareto_df = pareto_to_df(pareto)

        if pareto_df is None or len(pareto_df) == 0:
            st.error("Geen haalbare route gevonden binnen constraints")
            for k in STATE_KEYS:
                ss.pop(k, None)
        else:
            route_data, _, _ = select_best_pareto_route(
                pareto, min_dist, max_dist, max_bankjes_afstand
            )

            ss["bankjes_df"] = pareto_df
            ss["bankjes_weights"] = weights

            # standaardkeuze = de door select_best_pareto_route gekozen route
            default_path = list(route_data[6])
            matches = pareto_df.index[
                pareto_df["route"].apply(lambda r: list(r) == default_path)
            ]
            ss["bankjes_pos"] = int(matches[0]) if len(matches) else 0
            ss["bankjes_cid"] = ss.get("bankjes_cid", 0) + 1

    # ---- Render op basis van de geselecteerde route in session_state ----
    df_route = None
    bankjes_gdf = None
    route = False
    distance = 0
    score = 0
    max_gap = 0
    pareto_df = ss.get("bankjes_df")

    if pareto_df is not None and len(pareto_df) > 0:
        current_weights = ss["bankjes_weights"]
        pos = ss.get("bankjes_pos")
        if pos is not None and 0 <= pos < len(pareto_df):
            row = pareto_df.iloc[pos]
            df_route = pareto_path_to_gdf(gdf, row["route"])
            bankjes_gdf = _bankjes_voor_route(tuple(row["route"]), gdf)
            distance = row["afstand"]
            score = row["gemiddelde_score"]
            max_gap = row["max_gap"]
            route = True
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
            feature_group_to_add=route_layer(df_route, bankjes_gdf) if route else folium.FeatureGroup(name="Route"),
            width=700, height=620, returned_objects=[], key="bankjes_map",
        )

    with col_side:
        if pareto_df is not None and len(pareto_df) > 0:
            if route:
                m1, m2, m3 = st.columns(3)
                m1.metric("Afstand (km)", f"{distance / 1000:.2f}")
                m2.metric("Score", f"{score:.2f}")
                m3.metric("Langste stuk (m)", f"{max_gap:.0f}")
                if score == -10:
                    st.warning("Niet mogelijk om alle waypoints te bezoeken")

            st.markdown("**Alternatieve routes op het Pareto-front**")
            st.caption("Klik op een punt om die route op de kaart te zien.")
            fig = plot_pareto(
                pareto_df,
                selected_gap=max_gap if route else None,
                selected_score=score if route else None,
                selected_distance=distance if route else None,
            )
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"bankjes_pareto_{ss.get('bankjes_cid', 0)}",
                on_select="rerun",
                selection_mode="points",
            )

            clicked = clicked_point_index(event, len(pareto_df))
            if clicked is not None and clicked != ss.get("bankjes_pos"):
                ss["bankjes_pos"] = clicked
                st.rerun()
        else:
            st.markdown("### Zo begin je")
            st.markdown("""
**1.** Stel links de gewichten in en klik **Bereken** — de kaart kleurt mee.

**2.** Zoek op de kaart je start- en eindknooppunt (de zwarte punten).

**3.** Vul beide nummers links in, kies hoe ver je maximaal tussen twee bankjes
wilt lopen, en klik **Route toevoegen**.
            """)
            st.info(
                "Na het berekenen verschijnt hier een grafiek met alternatieve routes. "
                "Klik op een punt om die route op de kaart te bekijken.",
                icon="📊",
            )


# Run the app
if __name__ == '__main__':
    main()
