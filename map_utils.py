"""Snelle, herbruikbare basiskaart voor de loopplezier-pagina's.

De achtergrond (het gekleurde netwerk) hangt alleen af van de gekozen
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
from branca.element import MacroElement
from jinja2 import Template

N_BUCKETS = 11

# Standaardgewichten zoals in load_data() van elke pagina (voor de kleuring
# voordat de gebruiker iets berekent).
DEFAULT_WEIGHTS = dict(
    ovl=0, bomen=0, water=1, monumenten=0, wegen=-1, parken=3, toiletten=0,
    verkeerslichten=0, wegdekkwaliteit=0, horeca=0, kerk=0, winkels=0,
    groen=0, kampioen=0, waarnemingen=0, ov=0, schaduw=0,
)


def weights_key(weights):
    """Hashbare sleutel voor de cache (volgorde-onafhankelijk)."""
    return tuple(sorted((k, float(v)) for k, v in weights.items()))


def weight_form(form_key="Score input"):
    """Gewicht-inputs in de sidebar, gegroepeerd in drie inklapbare secties.

    Geeft ``(weights, calculate_button)`` terug. Schuifjes in plaats van
    getalvelden: die werken met een vinger en roepen op een smartboard geen
    schermtoetsenbord op. Door de twaalf factoren in te klappen past de hele
    sidebar op één scherm.
    """
    st.sidebar.header("Omgevingsfactoren")
    st.sidebar.caption("Geef elke factor een gewicht van –10 tot +10.")

    with st.sidebar.form(form_key):
        with st.expander("Groen & natuur"):
            bomen = st.slider("Bomen", -10, 10, 0, 1, key="bomen")
            parken = st.slider("Parken", -10, 10, 3, 1, key="parken")
            groen = st.slider("Groen", -10, 10, 0, 1, key="groen")
            water = st.slider("Water", -10, 10, 1, 1, key="water")
            schaduw = st.slider("Schaduw", -10, 10, 0, 1, key="schaduw")

        with st.expander("Voorzieningen"):
            horeca = st.slider("Horeca", -10, 10, 0, 1, key="horeca")
            winkels = st.slider("Winkels", -10, 10, 0, 1, key="winkels")
            ov = st.slider("Openbaar vervoer", -10, 10, 0, 1, key="ov")
            monumenten = st.slider("Monumenten", -10, 10, 0, 1, key="monumenten")

        with st.expander("Veiligheid & comfort"):
            ovl = st.slider("Openbare verlichting", -10, 10, 0, 1, key="ovl")
            verkeerslichten = st.slider("Verkeerslichten", -10, 10, 0, 1, key="verkeerslichten")
            wegen = st.slider("Drukke wegen", -10, 10, -1, 1, key="wegen")

        calculate_button = st.form_submit_button("Bereken", use_container_width=True)

    weights = dict(
        ovl=ovl, bomen=bomen, water=water, monumenten=monumenten,
        wegen=wegen, parken=parken, toiletten=0, verkeerslichten=verkeerslichten,
        wegdekkwaliteit=0, horeca=horeca, kerk=0, winkels=winkels,
        groen=groen, kampioen=0, waarnemingen=0, ov=ov, schaduw=schaduw,
    )
    return weights, calculate_button


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


class TouchMap(MacroElement):
    """Maakt de Leaflet-kaart prettig met een vinger op een groot touchscreen."""

    _template = Template("""
        {% macro script(this, kwargs) %}
        // Een vinger trilt tijdens een tik; zonder speling telt dat als slepen.
        L.Draggable.prototype.options.clickTolerance = 15;

        // st_folium meldt alleen een gewijzigde klik. Een verschil van een tiende
        // millimeter zorgt dat ook een tweede tik op dezelfde plek doorkomt.
        {{ this._parent.get_name() }}.on('click', function (event) {
            event.latlng.lat += Math.random() * 1e-9;
        });

        var touchStyle = document.createElement('style');
        touchStyle.textContent =
            '.leaflet-bar a, .leaflet-touch .leaflet-bar a' +
            '{width:max(56px,5vw);height:max(56px,5vw);line-height:max(56px,5vw);' +
            'font-size:max(32px,3vw);}' +
            '#map_div{height:100vh !important;}';
        document.head.appendChild(touchStyle);

        // De hoogte komt uit de CSS van de pagina: houd Leaflet daarmee in de pas.
        new ResizeObserver(function () {
            {{ this._parent.get_name() }}.invalidateSize();
        }).observe(document.body);
        {% endmacro %}
    """)


@st.cache_resource(show_spinner=False)
def _build_base_map(_gdf, wkey):
    """Bouw de zware basiskaart één keer per gewichten-set (wkey).

    ``_gdf`` wordt door de underscore niet gehasht; de cache-sleutel is enkel
    ``wkey``. De aanroeper zorgt dat ``_gdf['score_totaal']`` bij die gewichten
    hoort voordat een nieuwe sleutel voor het eerst wordt gebouwd.
    """
    center = [_gdf.geometry.centroid.y.mean(), _gdf.geometry.centroid.x.mean()]
    # prefer_canvas: teken het netwerk op een canvas i.p.v. losse SVG-paths ->
    # veel sneller renderen en pannen in de browser.
    # double_click_zoom uit: twee snelle tikken zijn twee gekozen punten.
    m = folium.Map(location=center, zoom_start=14, prefer_canvas=True, double_click_zoom=False)

    folium.GeoJson(
        _bucketed_geojson(_gdf),
        name="score_totaal",
        style_function=lambda f: {"color": f["properties"]["color"]},
    ).add_to(m)

    TouchMap().add_to(m)
    return m


def base_map(_gdf, weights):
    """Geef een verse kopie van de (gecachte) basiskaart voor deze gewichten.

    De zware achtergrond (gebucketed netwerk) wordt één keer per gewichten-set
    gebouwd; per rerun maken we er enkel een goedkope deep-copy van. We geven
    bewust een KOPIE terug omdat st_folium intern
    ``feature_group_to_add.add_to(map)`` doet en de meegegeven kaart dus muteert
    -- zo blijft het gecachte origineel schoon (geen oude routes), terwijl de
    achtergrond-string identiek blijft zodat st_folium enkel de routelaag
    bijwerkt i.p.v. het hele netwerk opnieuw te tekenen.
    """
    return copy.deepcopy(_build_base_map(_gdf, weights_key(weights)))
