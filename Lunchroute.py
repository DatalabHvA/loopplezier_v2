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
from streamlit_plotly_events import plotly_events
st.set_page_config(layout = 'wide')

@st.cache_data
def load_data():
	gdf = gpd.read_feather('./data/gdf.feather')
	gdf = gdf.reset_index(drop = True)
	gdf = calculate_new_column(gdf, ovl = 0, bomen = 1, water = -1, monumenten = 0, wegen = 0, parken = 0, toiletten = 0, verkeerslichten = -1, wegdekkwaliteit = 0, horeca = 1, kerk = 0, winkels = 0, groen = 0, kampioen = 0, waarnemingen = 0,ov = 1, schaduw = 0)
	
	nodes = gpd.read_feather('./data/nodes.feather').to_crs('EPSG:4326')
	nodes = nodes.reset_index().rename(columns = {'osmid' : 'knooppunt'})
	return (gdf, nodes)

def style_function(feature):
	cmap = cm.RdYlGn  # Choose a continuous colormap (11 colors)
	value = feature['properties']['score_totaal']  # Get the value from your column
	normalized_value = (0.5*value)+0.5
	color = mcolors.rgb2hex(cmap(normalized_value))  # Map the value to a color
	return {'color': color}
	
def style_function_route(feature):
	return {'weight': 5}

def calculate_new_column(gdf, ovl, bomen, water, monumenten, wegen, parken, toiletten, verkeerslichten, wegdekkwaliteit, horeca, kerk, winkels, groen, kampioen, waarnemingen, ov, schaduw, colum_name = 'Score'): 
    # Add your calculation logic here, e.g., using min_value and max_value
	# Score op basis van gewichten ingevuld op streamlit
	gdf['score_totaal'] = (
					gdf['score_bomen']*bomen + 
					gdf['score_ovl']*ovl + 
					gdf['score_water']*water + 
					gdf['score_monumenten']*monumenten + 
					gdf['score_wegen']*wegen + 
					gdf['score_park']*parken + 
					#gdf['score_openbare_toiletten']*toiletten + 
					gdf['score_verkeerslichten']*verkeerslichten + 
					#gdf['score_wegdekkwaliteit']*wegdekkwaliteit + 
					gdf['score_horeca']*horeca +
					#gdf['score_kerk']*kerk +
					gdf['score_OV']*ov +
					#gdf['score_buffergebied']*groen +
					#gdf['score_kampioensbomen']*kampioen +
					gdf['score_schaduw']*schaduw +
					gdf['score_winkels']*winkels) 
	return gdf	

#@st.cache_resource
def create_map(_gdf, _nodes, _df_route = None, route = False, distance = 0, score = 0):
	m = folium.Map(location=[_gdf['geometry'].centroid.y.mean(), _gdf['geometry'].centroid.x.mean()], zoom_start=14)

	folium.GeoJson(
		_gdf,
		name='score_totaal',
		style_function=style_function).add_to(m)	
	
	folium.GeoJson(
		_nodes,
		name='Nodes',
		marker = folium.CircleMarker(radius = 2, # Radius in metres
                                           weight = 0, #outline weight
                                           fill_color = '#000000', 
                                           fill_opacity = 1),
		tooltip=folium.GeoJsonTooltip(fields=['knooppunt'], labels=True, sticky=True)
		).add_to(m)	
	
	if route: 
		folium.GeoJson(
			_df_route, style_function=style_function_route,
			name='Route').add_to(m)	
		st.markdown('**Er is een route gevonden van '+str(round(distance/1000,2))+ 'km en een gemiddelde score van '+str(round(score,2))
			  		+ '**' )
		if score == -10: ###
			st.markdown("Niet mogelijk om alle waypoints te bezoeken") ###
    
	if route and _df_route is not None and len(_df_route) > 0:

		route_gdf = _df_route[_df_route.geometry.notna()].copy()

		# 🔥 FORCE correct CRS voor route
		if route_gdf.crs is None:
			route_gdf = route_gdf.set_crs("EPSG:4326")
	return m




st.title("Loopplezierkaart!!!")
st.write("Welkom bij de loopplezierkaart van de Hogeschool van Amsterdam. Kies in het menu links welke omgevingsfactoren je wil laten meewegen in de loopbaarheidsscore. Klik op 'Calculate' en bekijk de kaart (groen is aantrekkelijk, geel is neutraal en rood is minder aantrekkelijk).")
st.write("Je kan waarnemingen van dieren en planten op de kaart tonen door het vinkje aan te zetten. Bekijk zo wat je onderweg allemaal tegenkomt.")
st.write("De zwarte punten zijn knooppunten met een id. Als je twee knooppunten kiest en de id's invult in het menu kan je ook de meest aantrekkelijke route bereken tussen de twee punten a.d.v. de eerder gekozen score. Klik op 'Add route' om jouw gepersonaliseerde route te tonen")
	
(gdf, nodes) = load_data()
	
	# Sidebar with sliders
st.sidebar.header("Map settings")

df_route = None
	# df_route = []
route = False
distance = 0
score = 0
pareto_df = None

	
with st.sidebar.form("Score input"):	
	ovl = st.number_input("Score openbare verlichting", -10,10,0,1,  key="ovl")
	bomen = st.number_input("Score bomen", -10,10,1,1, key="bomen")
	water = st.number_input("Score water", -10,10,-1,1, key="water")
	monumenten = st.number_input("Score monumenten", -10,10,0,1, key="monumenten")
	wegen = st.number_input("Score drukke wegen", -10,10,0,1, key="wegen")
	parken = st.number_input("Score parken", -10,10,0,1, key="parken")
	#toiletten = st.number_input("Score toiletten", -10,10,0,1, key="toiletten")
	verkeerslichten = st.number_input("Score verkeerslichten", -10,10,-1,1, key="verkeerslichten")
	#wegdekkwaliteit = st.number_input("Score wegdekkwaliteit", -10,10,0,1, key="wegdekkwaliteit")
	horeca = st.number_input("Score horeca", -10,10,1,1, key="horeca")
	#kerk = st.number_input("Score kerken", -10,10,0,1, key="kerk")
	winkels = st.number_input("Score winkels", -10,10,0,1, key="winkels")
	groen = st.number_input("Score groen", -10,10,0,1, key="groen")
	# kampioen = st.number_input("Score kampioensbomen", -10,10,0,1, key="kampioen")
	schaduw = st.number_input("Score schaduw", -10, 10, 0, 1, key = "schaduw")
	# waarnemingen = st.number_input("Score waarnemingen", -10,10,2,1, key="waarnemingen")
	ov = st.number_input("Score OV", -10,10,1,1, key="ov")
	calculate_button = st.form_submit_button("Calculate")

with st.sidebar.form("Route"):	
	start = st.number_input("Start knooppunt", 0,3100,924,1,  key="start")
	end = st.number_input("Eind knooppunt", 0,3100,1145,1,  key="end")
	min_dist = st.number_input("Minimale afstand", 500,10000,500,100,  key="min_dist")
	max_dist = st.number_input("Maximale afstand", 500,10000,3000,100,  key="max_dist")
	add_route = st.form_submit_button("Add route")
			
if calculate_button:
	gdf = calculate_new_column(gdf, ovl = ovl, bomen = bomen, water = water, monumenten = monumenten, wegen = wegen, parken = parken, toiletten = 0, verkeerslichten = verkeerslichten, wegdekkwaliteit = 0, horeca = horeca, kerk = 0, winkels = winkels ,groen = groen, kampioen = 0, waarnemingen = 0, ov = ov, schaduw = schaduw)

if add_route:

    gdf = calculate_new_column(
    gdf,
    ovl=ovl,
    bomen=bomen,
    water=water,
    monumenten=monumenten,
    wegen=wegen,
    parken=parken,
    toiletten=0,
    verkeerslichten=verkeerslichten,
    wegdekkwaliteit=0,
    horeca=horeca,
    kerk=0,
    winkels=winkels,
    groen=groen,
    kampioen=0,
    waarnemingen=0,
    ov=ov,
    schaduw=schaduw
)
    pareto = generate_pareto_routes_2d(
        gdf,
        start=start,
        end=end,
        L_min=min_dist,
        L_max=max_dist,
        max_lens=[10,25,50,100,250]
    )
    pareto_df = labels_to_dataframe(pareto)

    if len(pareto) == 0:

        st.error("Geen routes gevonden")

    else:

        best_route = max(
            pareto,
            key=lambda x: x[2] / x[1]
        )

        path = best_route[3]

        distance = best_route[1]

        score = best_route[2] / best_route[1]

        df_route = pareto_path_to_gdf(
            gdf,
            path
        )

        route = True

st_folium(create_map(gdf, nodes, df_route, route, distance, score), width=1000, height=700, returned_objects=[])
if pareto_df is not None and len(pareto_df) > 0:

	st.plotly_chart(plot_pareto_2d(pareto_df, selected_score=score, selected_distance=distance), use_container_width = True)


route = False
	