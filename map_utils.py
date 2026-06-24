"""Snelle, herbruikbare basiskaart voor de loopplezier-pagina's.

De achtergrond (gekleurd netwerk + knooppunten) hangt alleen af van de gekozen
gewichten, niet van welke route je aanklikt. Daarom bouwen we hem één keer per
gewichten-set (``@st.cache_resource``) en maken we er per rerun alleen een
goedkope deep-copy van. Het netwerk (9852 edges) wordt bovendien samengevoegd
tot ``N_BUCKETS`` kleurklassen, zodat zowel de Python-build als het tekenen in
de browser vele malen sneller wordt.
"""
import copy

import numpy as np
import folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import streamlit as st

N_BUCKETS = 11

# Standaardgewichten zoals in load_data() van elke pagina (voor de kleuring
# voordat de gebruiker iets berekent).
DEFAULT_WEIGHTS = dict(
    ovl=0, bomen=1, water=-1, monumenten=0, wegen=0, parken=0, toiletten=0,
    verkeerslichten=-1, wegdekkwaliteit=0, horeca=1, kerk=0, winkels=0,
    groen=0, kampioen=0, waarnemingen=0, ov=1, schaduw=0,
)


def weights_key(weights):
    """Hashbare sleutel voor de cache (volgorde-onafhankelijk)."""
    return tuple(sorted((k, float(v)) for k, v in weights.items()))


def _bucketed_geojson(_gdf):
    """Voeg edges samen tot N_BUCKETS kleurklassen en geef GeoJSON terug."""
    cmap = cm.RdYlGn
    norm = np.clip(0.5 * _gdf["score_totaal"].to_numpy(dtype=float) + 0.5, 0, 1)
    bucket = np.round(norm * (N_BUCKETS - 1)).astype(int)

    g = _gdf[["geometry"]].copy()
    g["bucket"] = bucket
    diss = g.dissolve(by="bucket")
    diss["color"] = [mcolors.rgb2hex(cmap(b / (N_BUCKETS - 1))) for b in diss.index]
    return diss.to_json()


@st.cache_resource(show_spinner=False)
def _build_base_map(_gdf, _nodes, wkey):
    """Bouw de zware basiskaart één keer per gewichten-set (wkey).

    ``_gdf``/``_nodes`` worden door de underscore niet gehasht; de cache-sleutel
    is enkel ``wkey``. De aanroeper zorgt dat ``_gdf['score_totaal']`` bij die
    gewichten hoort voordat een nieuwe sleutel voor het eerst wordt gebouwd.
    """
    center = [_gdf.geometry.centroid.y.mean(), _gdf.geometry.centroid.x.mean()]
    # prefer_canvas: teken de duizenden knooppunten/lijnen op een canvas i.p.v.
    # losse SVG-paths -> veel sneller renderen en pannen in de browser.
    m = folium.Map(location=center, zoom_start=14, prefer_canvas=True)

    folium.GeoJson(
        _bucketed_geojson(_gdf),
        name="score_totaal",
        style_function=lambda f: {"color": f["properties"]["color"]},
    ).add_to(m)

    nodes_min = _nodes[["knooppunt", "geometry"]] if "knooppunt" in _nodes.columns else _nodes
    folium.GeoJson(
        nodes_min,
        name="Nodes",
        marker=folium.CircleMarker(radius=2, weight=0, fill_color="#000000", fill_opacity=1),
        tooltip=folium.GeoJsonTooltip(fields=["knooppunt"], labels=True, sticky=True),
    ).add_to(m)

    return m


def base_map(_gdf, _nodes, weights):
    """Geef een verse kopie van de (gecachte) basiskaart voor deze gewichten.

    De zware achtergrond (gebucketed netwerk + knooppunten) wordt één keer per
    gewichten-set gebouwd; per rerun maken we er enkel een goedkope deep-copy
    van. We geven bewust een KOPIE terug omdat st_folium intern
    ``feature_group_to_add.add_to(map)`` doet en de meegegeven kaart dus muteert
    -- zo blijft het gecachte origineel schoon (geen oude routes), terwijl de
    achtergrond-string identiek blijft zodat st_folium enkel de routelaag
    bijwerkt i.p.v. het hele netwerk + alle knooppunten opnieuw te tekenen.
    """
    return copy.deepcopy(_build_base_map(_gdf, _nodes, weights_key(weights)))
