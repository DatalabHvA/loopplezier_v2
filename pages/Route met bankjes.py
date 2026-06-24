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
from map_utils import base_map, DEFAULT_WEIGHTS, weights_key
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
    gdf = calculate_new_column(gdf, ovl=0, bomen=1, water=-1, monumenten=0, wegen=0, parken=0, toiletten=0, verkeerslichten=-1, wegdekkwaliteit=0, horeca=1, kerk=0, winkels=0, groen=0, kampioen=0, waarnemingen=0, ov=1, schaduw=0)

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

    st.title("Loopplezierkaart")
    st.write("Welkom bij de loopplezierkaart van de Hogeschool van Amsterdam. Kies in het menu links welke omgevingsfactoren je wil laten meewegen in de loopbaarheidsscore. Klik op 'Calculate' en bekijk de kaart (groen is aantrekkelijk, geel is neutraal en rood is minder aantrekkelijk).")
    st.write("Je kan waarnemingen van dieren en planten op de kaart tonen door het vinkje aan te zetten. Bekijk zo wat je onderweg allemaal tegenkomt.")
    st.write("De zwarte punten zijn knooppunten met een id. Als je twee knooppunten kiest en de id's invult in het menu kan je ook de meest aantrekkelijke route bereken tussen de twee punten a.d.v. de eerder gekozen score. Klik op 'Add route' om jouw gepersonaliseerde route te tonen")

    (gdf, nodes) = load_data()

    # Sidebar with sliders
    st.sidebar.header("Map settings")

    with st.sidebar.form("Score input"):
        ovl = st.number_input("Score openbare verlichting", -10, 10, 0, 1, key="ovl")
        bomen = st.number_input("Score bomen", -10, 10, 1, 1, key="bomen")
        water = st.number_input("Score water", -10, 10, -1, 1, key="water")
        monumenten = st.number_input("Score monumenten", -10, 10, 0, 1, key="monumenten")
        wegen = st.number_input("Score drukke wegen", -10, 10, 0, 1, key="wegen")
        parken = st.number_input("Score parken", -10, 10, 0, 1, key="parken")
        verkeerslichten = st.number_input("Score verkeerslichten", -10, 10, -1, 1, key="verkeerslichten")
        horeca = st.number_input("Score horeca", -10, 10, 1, 1, key="horeca")
        winkels = st.number_input("Score winkels", -10, 10, 0, 1, key="winkels")
        groen = st.number_input("Score groen", -10, 10, 0, 1, key="groen")
        schaduw = st.number_input("Score schaduw", -10, 10, 0, 1, key="schaduw")
        ov = st.number_input("Score OV", -10, 10, 1, 1, key="ov")
        calculate_button = st.form_submit_button("Calculate")

    with st.sidebar.form("Route"):
        start = st.number_input("Start knooppunt", 0, 3100, 924, 1, key="start")
        end = st.number_input("Eind knooppunt", 0, 3100, 1145, 1, key="end")
        min_dist = st.number_input("Minimale afstand", 500, 10000, 500, 100, key="min_dist")
        max_dist = st.number_input("Maximale afstand", 500, 10000, 3000, 100, key="max_dist")
        max_bankjes_afstand = st.number_input("Max afstand tussen bankje", 100, 2000, 500, 50)
        add_route = st.form_submit_button("Add route")

    weights = dict(
        ovl=ovl, bomen=bomen, water=water, monumenten=monumenten,
        wegen=wegen, parken=parken, toiletten=0, verkeerslichten=verkeerslichten,
        wegdekkwaliteit=0, horeca=horeca, kerk=0, winkels=winkels,
        groen=groen, kampioen=0, waarnemingen=0, ov=ov, schaduw=schaduw,
    )

    ss = st.session_state
    STATE_KEYS = ("bankjes_df", "bankjes_weights", "bankjes_pos", "bankjes_cid")

    # 'Calculate' = alleen de kaart herkleuren; verwijder een eventuele actieve route
    if calculate_button:
        gdf = calculate_new_column(gdf, **weights)
        for k in STATE_KEYS:
            ss.pop(k, None)

    # 'Add route' = nieuwe Pareto-set berekenen en bewaren in session_state
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
    st_folium(
        base_map(gdf, nodes, current_weights),
        feature_group_to_add=route_layer(df_route, bankjes_gdf) if route else None,
        width=1000, height=700, returned_objects=[], key="bankjes_map",
    )
    if route:
        st.markdown('**Er is een route gevonden van ' + str(round(distance / 1000, 2)) + 'km met een max gap van ' + str(round(max_gap, 0)) + 'm en een gemiddelde score van ' + str(round(score, 2)) + '**')
        if score == -10:
            st.markdown("Niet mogelijk om alle waypoints te bezoeken")

    if pareto_df is not None and len(pareto_df) > 0:
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
        st.caption("Klik op een punt in de grafiek om die alternatieve route (bv. kleinere afstand tussen bankjes met een lagere score, maar op de Pareto-front) op de kaart te zien.")

        clicked = clicked_point_index(event, len(pareto_df))
        if clicked is not None and clicked != ss.get("bankjes_pos"):
            ss["bankjes_pos"] = clicked
            st.rerun()


# Run the app
if __name__ == '__main__':
    main()
