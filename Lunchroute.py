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
def _cached_pareto_2d(_gdf, start, end, L_min, L_max, wkey):
    """Cache Pareto-routes per combinatie van start/eind/afstand/gewichten."""
    return generate_pareto_routes_2d(_gdf, start, end, L_min, L_max)


@st.cache_data
def load_data():
    gdf = gpd.read_feather('./data/gdf.feather')
    gdf = gdf.reset_index(drop=True)
    gdf = calculate_new_column(gdf, ovl=0, bomen=1, water=-1, monumenten=0, wegen=0, parken=0, toiletten=0, verkeerslichten=-1, wegdekkwaliteit=0, horeca=1, kerk=0, winkels=0, groen=0, kampioen=0, waarnemingen=0, ov=1, schaduw=0)

    nodes = gpd.read_feather('./data/nodes.feather').to_crs('EPSG:4326')
    nodes = nodes.reset_index().rename(columns={'osmid': 'knooppunt'})
    return (gdf, nodes)


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


def route_layer(_df_route):
    """FeatureGroup met enkel de route; wordt los van de (statische) basiskaart
    bijgewerkt via st_folium(feature_group_to_add=...), zodat een klik op de
    grafiek niet het hele netwerk + alle knooppunten opnieuw rendert."""
    fg = folium.FeatureGroup(name='Route')
    for geom in _df_route.geometry:
        if geom is None:
            continue
        parts = geom.geoms if geom.geom_type == 'MultiLineString' else [geom]
        for part in parts:
            folium.PolyLine([(y, x) for x, y in part.coords], color='#3388ff', weight=5).add_to(fg)
    return fg


st.title("Loopplezierkaart")
st.caption("Hogeschool van Amsterdam — Verken en vergelijk wandelroutes op basis van jouw persoonlijke score")

(gdf, nodes) = load_data()

# Sidebar
st.sidebar.header("Omgevingsfactoren")
st.sidebar.caption("Gebruik –10 tot +10 om factoren mee te laten wegen in de routescore.")

with st.sidebar.form("Score input"):
    ovl = st.number_input("Openbare verlichting", -10, 10, 0, 1, key="ovl")
    bomen = st.number_input("Bomen", -10, 10, 1, 1, key="bomen")
    water = st.number_input("Water", -10, 10, -1, 1, key="water")
    monumenten = st.number_input("Monumenten", -10, 10, 0, 1, key="monumenten")
    wegen = st.number_input("Drukke wegen", -10, 10, 0, 1, key="wegen")
    parken = st.number_input("Parken", -10, 10, 0, 1, key="parken")
    verkeerslichten = st.number_input("Verkeerslichten", -10, 10, -1, 1, key="verkeerslichten")
    horeca = st.number_input("Horeca", -10, 10, 1, 1, key="horeca")
    winkels = st.number_input("Winkels", -10, 10, 0, 1, key="winkels")
    groen = st.number_input("Groen", -10, 10, 0, 1, key="groen")
    schaduw = st.number_input("Schaduw", -10, 10, 0, 1, key="schaduw")
    ov = st.number_input("Openbaar vervoer", -10, 10, 1, 1, key="ov")
    calculate_button = st.form_submit_button("Calculate", use_container_width=True)

st.sidebar.divider()
st.sidebar.subheader("Route berekenen")

with st.sidebar.form("Route"):
    start = st.number_input("Start knooppunt", 0, 3100, 924, 1, key="start")
    end = st.number_input("Eind knooppunt", 0, 3100, 1145, 1, key="end")
    min_dist = st.number_input("Minimale afstand (m)", 500, 10000, 500, 100, key="min_dist")
    max_dist = st.number_input("Maximale afstand (m)", 500, 10000, 3000, 100, key="max_dist")
    add_route = st.form_submit_button("Add route", use_container_width=True)

weights = dict(
    ovl=ovl, bomen=bomen, water=water, monumenten=monumenten,
    wegen=wegen, parken=parken, toiletten=0, verkeerslichten=verkeerslichten,
    wegdekkwaliteit=0, horeca=horeca, kerk=0, winkels=winkels,
    groen=groen, kampioen=0, waarnemingen=0, ov=ov, schaduw=schaduw,
)

ss = st.session_state
STATE_KEYS = ("home_df", "home_weights", "home_pos", "home_cid")

# 'Calculate' = alleen de kaart herkleuren; verwijder een eventuele actieve route
if calculate_button:
    gdf = calculate_new_column(gdf, **weights)
    for k in STATE_KEYS:
        ss.pop(k, None)

# 'Add route' = nieuwe Pareto-set berekenen en bewaren in session_state
if add_route:
    gdf = calculate_new_column(gdf, **weights)
    pareto = _cached_pareto_2d(
        gdf, start=start, end=end, L_min=min_dist, L_max=max_dist,
        wkey=weights_key(weights)
    )
    pareto_df = labels_to_dataframe(pareto)

    if len(pareto_df) == 0:
        st.error("Geen routes gevonden")
        for k in STATE_KEYS:
            ss.pop(k, None)
    else:
        ss["home_df"] = pareto_df
        ss["home_weights"] = weights
        # standaardkeuze = route met de hoogste gemiddelde score (zoals voorheen)
        ss["home_pos"] = int(pareto_df["gemiddelde_score"].values.argmax())
        ss["home_cid"] = ss.get("home_cid", 0) + 1

# ---- Render op basis van de geselecteerde route in session_state ----
df_route = None
route = False
distance = 0
score = 0
pareto_df = ss.get("home_df")

if pareto_df is not None and len(pareto_df) > 0:
    current_weights = ss["home_weights"]
    pos = ss.get("home_pos")
    if pos is not None and 0 <= pos < len(pareto_df):
        row = pareto_df.iloc[pos]
        df_route = pareto_path_to_gdf(gdf, row["route"])
        distance = row["afstand"]
        score = row["gemiddelde_score"]
        route = True
elif calculate_button:
    current_weights = weights
else:
    current_weights = DEFAULT_WEIGHTS

gdf = calculate_new_column(gdf, **current_weights)

col_map, col_side = st.columns([3, 2], gap="medium")

with col_map:
    st_folium(
        base_map(gdf, nodes, current_weights),
        feature_group_to_add=route_layer(df_route) if route else None,
        width=700, height=620, returned_objects=[], key="home_map",
    )

with col_side:
    if pareto_df is not None and len(pareto_df) > 0:
        if route:
            m1, m2 = st.columns(2)
            m1.metric("Afstand", f"{distance / 1000:.2f} km")
            m2.metric("Score", f"{score:.2f}")
            if score == -10:
                st.warning("Niet mogelijk om alle waypoints te bezoeken")

        st.markdown("**Alternatieve routes op het Pareto-front**")
        st.caption("Klik op een punt om die route op de kaart te zien.")
        fig = plot_pareto_2d(
            pareto_df,
            selected_score=score if route else None,
            selected_distance=distance if route else None,
        )
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"home_pareto_{ss.get('home_cid', 0)}",
            on_select="rerun",
            selection_mode="points",
        )

        clicked = clicked_point_index(event, len(pareto_df))
        if clicked is not None and clicked != ss.get("home_pos"):
            ss["home_pos"] = clicked
            st.rerun()
    else:
        st.markdown("### Hoe werkt het?")
        st.markdown("""
**1. Stel gewichten in** via het menu links
Kies welke omgevingsfactoren je belangrijk vindt — positief of negatief.

**2. Klik Calculate**
De kaart kleurt op basis van jouw score:
🟢 groen = aantrekkelijk &nbsp;·&nbsp; 🟡 geel = neutraal &nbsp;·&nbsp; 🔴 rood = minder

**3. Kies twee knooppunten**
De zwarte punten op de kaart hebben elk een nummer.
Noteer het nummer van je start- en eindpunt.

**4. Klik Add route**
Het algoritme berekent routes op het **Pareto-front**: optimaal in zowel score als afstand.
Hier verschijnt dan een interactieve grafiek — klik op elk punt om
die route direct op de kaart te bekijken.
        """)
