import streamlit as st
import pandas as pd
import geopandas as gpd
from streamlit_folium import st_folium
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from functions_shadow import *
import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import linemerge
from pareto_select import clicked_point_index
from map_utils import base_map, DEFAULT_WEIGHTS, weights_key, weight_form
st.set_page_config(layout='wide')


@st.cache_data(show_spinner="Routes berekenen...")
def _cached_pareto_shadow(_gdf, start, end, L_min, L_max, wkey):
    """Cache schaduw-Pareto-routes per combinatie van start/eind/afstand/gewichten."""
    return generate_pareto_routes_2d(_gdf, start, end, L_min, L_max)


@st.cache_data
def load_data():
    gdf = gpd.read_feather('./data/gdf.feather')
    gdf = gdf.reset_index(drop=True)
    gdf = calculate_new_column(gdf, **DEFAULT_WEIGHTS)

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
    bijgewerkt via st_folium(feature_group_to_add=...)."""
    fg = folium.FeatureGroup(name='Route')
    for geom in _df_route.geometry:
        if geom is None:
            continue
        parts = geom.geoms if geom.geom_type == 'MultiLineString' else [geom]
        for part in parts:
            folium.PolyLine([(y, x) for x, y in part.coords], color='#3388ff', weight=5).add_to(fg)
    return fg


def main():
    # Title and description

    st.title("Schaduwbedekking")
    st.caption("Hogeschool van Amsterdam — Vind een route die op warme dagen genoeg schaduw biedt")

    with st.expander("Over deze kaart — wat zie ik hier?"):
        st.markdown("""
Deze kaart zoekt wandelroutes met **zoveel mogelijk schaduw** — handig op warme dagen of
als je de zon liever mijdt.

**Jouw eigen score**
In het menu links geef je elke omgevingsfactor een gewicht tussen –10 en +10. Klik op
**Bereken** om de kaart te kleuren:
🟢 groen = aantrekkelijk &nbsp;·&nbsp; 🟡 geel = neutraal &nbsp;·&nbsp; 🔴 rood = minder aantrekkelijk

**Een route berekenen**
Vul een start- en eindknooppunt in (de zwarte punten op de kaart; beweeg je muis over een
punt om het nummer te zien), stel in hoeveel schaduw je minimaal wilt, en klik op
**Route toevoegen**.

**Schaduw versus score, en het Pareto-front**
De schaduwrijkste route is zelden ook de aantrekkelijkste — meer schaduw gaat vaak ten
koste van de omgevingsscore. Er is dus geen enkele beste route. Het algoritme zoekt alle
routes die je niet kunt verbeteren zonder op de andere as in te leveren: het
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
        min_shadow = st.slider("Minimale schaduwbedekking (%)", 0, 100, 30)
        add_route = st.form_submit_button("Route toevoegen", use_container_width=True)

    ss = st.session_state
    STATE_KEYS = ("shadow_df", "shadow_weights", "shadow_pos", "shadow_cid")

    # 'Bereken' = alleen de kaart herkleuren; verwijder een eventuele actieve route
    if calculate_button:
        gdf = calculate_new_column(gdf, **weights)
        for k in STATE_KEYS:
            ss.pop(k, None)

    # 'Route toevoegen' = nieuwe Pareto-set berekenen en bewaren in session_state
    if add_route:
        gdf = calculate_new_column(gdf, **weights)

        pareto_labels = _cached_pareto_shadow(
            gdf, start=start, end=end, L_min=min_dist, L_max=max_dist,
            wkey=weights_key(weights)
        )

        pareto_df = pareto_to_df_2d(pareto_labels)

        if len(pareto_df) == 0:
            st.error("Geen routes gevonden")
            for k in STATE_KEYS:
                ss.pop(k, None)
        else:
            ss["shadow_df"] = pareto_df
            ss["shadow_weights"] = weights

            # standaardkeuze = beste score binnen de gewenste schaduw-constraint
            feasible = pareto_df[pareto_df["shadow_density"] >= (min_shadow / 100)]
            if len(feasible) == 0:
                st.error("Geen routes voldoen aan minimale schaduwbedekking")
                ss["shadow_pos"] = None
            else:
                ss["shadow_pos"] = int(feasible["score_density"].idxmax())
            ss["shadow_cid"] = ss.get("shadow_cid", 0) + 1

    # ---- Render op basis van de geselecteerde route in session_state ----
    df_route = None
    route = False
    distance = 0
    score = 0
    shadow_pct = 0
    pareto_df = ss.get("shadow_df")

    if pareto_df is not None and len(pareto_df) > 0:
        current_weights = ss["shadow_weights"]
        pos = ss.get("shadow_pos")
        if pos is not None and 0 <= pos < len(pareto_df):
            row = pareto_df.iloc[pos]
            df_route = pareto_path_to_gdf(gdf, row["path"])
            distance = row["afstand"]
            score = row["score_density"]
            shadow_pct = row["shadow_density"] * 100
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
            feature_group_to_add=route_layer(df_route) if route else folium.FeatureGroup(name="Route"),
            width=700, height=620, returned_objects=[], key="shadow_map",
        )

    with col_side:
        if pareto_df is not None and len(pareto_df) > 0:
            if route:
                m1, m2, m3 = st.columns(3)
                m1.metric("Afstand (km)", f"{distance / 1000:.2f}")
                m2.metric("Score", f"{score:.2f}")
                m3.metric("Schaduw", f"{shadow_pct:.0f}%")
                if score == -10:
                    st.warning("Niet mogelijk om alle waypoints te bezoeken")

            st.markdown("**Alternatieve routes op het Pareto-front**")
            st.caption("Klik op een punt om die route op de kaart te zien.")
            fig = plot_pareto(
                pareto_df,
                selected_shadow=shadow_pct if route else None,
                selected_score=score if route else None,
                selected_distance=distance if route else None,
            )
            event = st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"shadow_pareto_{ss.get('shadow_cid', 0)}",
                on_select="rerun",
                selection_mode="points",
            )

            clicked = clicked_point_index(event, len(pareto_df))
            if clicked is not None and clicked != ss.get("shadow_pos"):
                ss["shadow_pos"] = clicked
                st.rerun()
        else:
            st.markdown("### Zo begin je")
            st.markdown("""
**1.** Stel links de gewichten in en klik **Bereken** — de kaart kleurt mee.

**2.** Zoek op de kaart je start- en eindknooppunt (de zwarte punten).

**3.** Vul beide nummers links in, kies hoeveel schaduw je minimaal wilt,
en klik **Route toevoegen**.
            """)
            st.info(
                "Na het berekenen verschijnt hier een grafiek met alternatieve routes. "
                "Klik op een punt om die route op de kaart te bekijken.",
                icon="📊",
            )


# Run the app
if __name__ == '__main__':
    main()
