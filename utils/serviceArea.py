import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import osmnx as ox
import os
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
import shapely.ops as so
import pandas as pd
import numpy as np

class GetServiceArea:
    def __init__(self, G, bus_stops_df, train_exits_df):
        """ 
        Args:
            G (MultiDiGraph): graph of walking network
            train_exits_df (gpd): gpd of train_exits
            bus_stops_df (gpd): gpd of bus stops
        """
        self.G = G
        self.crs = bus_stops_df.crs
        # ensure CRS of input data are consistent
        if train_exits_df.crs != bus_stops_df.crs:
            train_exits_df = train_exits_df.to_crs(bus_stops_df.crs)
        self.bus_stops_df = bus_stops_df
        self.train_exits_df = train_exits_df

    def get_nearest_nodes(self,gdf):
        """ 
        Args:
            GeoDataFrame that has a geometry column to extract coordinates
        Returns:
            np.ndarray: nodes ID that may be in graph G
        """
        coords = gdf.geometry.get_coordinates()
        # x, y = lon, lat
        nearest_nodes = ox.distance.nearest_nodes(self.G, coords['x'], coords['y'])
        return nearest_nodes

    def get_publicTransport_nodes(self, bus_c='#f0f921',train_c='#0d0887',node_size=15, plot = True):
        """ 
        Args:
            bus_c (str): hex colour to colour bus nodes
            train_c (str): hex colour to colour train nodes
            node_size (float): size of nodes in G for plotting
            plot (bool): If True, plot the nearest bus stop and train exit nodes
        Returns:
            tuple of np.ndarray: bus stops and train exits nearest nodes in graph G
        """
        # get nearest nodes in graph G that is the closest to the locations of the bus stops and train exits
        bus_nodes = self.get_nearest_nodes(self.bus_stops_df)
        train_nodes = self.get_nearest_nodes(self.train_exits_df)
        if plot:
            # color the nodes according to isochrone then plot the street network
            node_colors = {}
            for node in bus_nodes:
                node_colors[node] = bus_c
            for node in train_nodes:
                node_colors[node] = train_c
            nc = [node_colors[node] if node in node_colors else "none" for node in self.G.nodes()]
            ns = [node_size if node in node_colors else 0 for node in self.G.nodes()]
            fig, ax = ox.plot_graph(
                self.G,
                node_color=nc,
                node_size=ns,
                node_alpha=0.8,
                edge_linewidth=0.2,
                edge_color="#999999",
            )
        return bus_nodes, train_nodes


    def get_serviceArea_nodes(self,bus_radius = 200, train_radius = 400, bus_c='#f0f921',train_c='#0d0887',node_size=15,plot = True):
        """ 
        Args:
            bus_radius (float): walking distance in metres from the nearest bus stops
            train_radius (float): walking distance in metres from the nearest train stops
            bus_c (str): hex colour to colour bus nodes
            train_c (str): hex colour to colour train nodes
            node_size (float): size of nodes in G for plotting
            plot (bool): plot graph and bus and train nodes
        Returns:
            list: identifies all the nodes that are within the bus_radius and train_radius using the service area method
        """
        # get nearest nodes in graph G that is the closest to the locations of the bus stops and train exits
        bus_nodes = self.get_nearest_nodes(self.bus_stops_df)
        train_nodes = self.get_nearest_nodes(self.train_exits_df)
        # color the nodes according to bus/train exits and color them separately
        node_colors = {}
        # identify all nodes within 200m of the bus stops
        for node in bus_nodes:
            try:
                subgraph_bus = nx.ego_graph(self.G, node, radius=bus_radius, distance="length")
                for n in subgraph_bus.nodes():
                    node_colors[n] = bus_c
            except Exception as e:
                pass
                # print(f'{node}: {e}')
        # identify all nodes within 400m of the train exits
        for node in train_nodes:
            try:
                subgraph_train = nx.ego_graph(self.G, node, radius=train_radius, distance="length")
                for n in subgraph_train.nodes():
                    node_colors[n] = train_c
            except Exception as e:
                pass
                # print(f'{node}: {e}')

        if plot:
            # colours of nodes
            nc = [node_colors[node] if node in node_colors else "none" for node in self.G.nodes()]
            # size of nodes
            ns = [node_size if node in node_colors else 0 for node in self.G.nodes()]
            fig, ax = ox.plot_graph(
                self.G,
                node_color=nc,
                node_size=ns,
                node_alpha=0.8,
                edge_linewidth=0.2,
                edge_color="#999999",
            )
        return list(node_colors)
    
    def make_convex_hull(self,subgraph):
        """ 
        Args:
            subgraph (MultiGraph): Graph
        Returns:
            Polygon that covers all the nodes in the subgraph, which means points and line strings are excluded and ignored
        """
        node_points = [Point((data["x"], data["y"])) for node, data in subgraph.nodes(data=True)]
        if len(node_points) <= 2:
            raise ValueError(f"Number of nodes less than 2: {len(node_points)}")
        bounding_poly = gpd.GeoSeries(node_points).unary_union.convex_hull
        return bounding_poly

    def get_serviceArea_polygons(self,bus_radius = 200, train_radius = 400):
        """ 
        Args:
            bus_radius (float): walking distance in metres from the nearest bus stops
            train_radius (float): walking distance in metres from the nearest train stops
        Returns:
            gpd.GeoDataFrame: polygon collection in a gdf, where each polygon represents the service area around each bus stop/train exit
        """
        # get nearest nodes in graph G that is the closest to the locations of the bus stops and train exits
        bus_nodes = self.get_nearest_nodes(self.bus_stops_df)
        train_nodes = self.get_nearest_nodes(self.train_exits_df)

        # store all the polygons
        isochrone_polys = []

        # identify all nodes within 200m of the bus stops
        for node in bus_nodes:
            try:
                subgraph_bus = nx.ego_graph(self.G, node, radius=bus_radius, distance="length")
                bus_poly = self.make_convex_hull(subgraph_bus)
                isochrone_polys.append(bus_poly)
                # break
            except Exception as e:
                pass
                # print(f'{node}: {e}')
        
        # identify all nodes within 200m of the bus stops
        for node in train_nodes:
            try:
                subgraph_train = nx.ego_graph(self.G, node, radius=train_radius, distance="length")
                train_poly = self.make_convex_hull(subgraph_train)
                isochrone_polys.append(train_poly)
                # break
            except Exception as e:
                pass
                # print(f'{node}: {e}')

        return gpd.GeoDataFrame(geometry=isochrone_polys)
    
    def plot_serviceArea(self,bus_radius = 200, train_radius = 400, bus_c='blue',train_c='red',serviceArea_c = '#f0f921',node_size=15,plot = True):
        """ 
        Args:
            bus_radius (float): walking distance in metres from the nearest bus stops
            train_radius (float): walking distance in metres from the nearest train stops
            bus_c (str): hex colour to colour bus nodes
            train_c (str): hex colour to colour train nodes
            node_size (float): size of nodes in G for plotting
            plot (bool): plot graph and bus and train nodes
        Returns:
            gpd.GeoDataFrame: polygon collection in a gdf that represents the union of the service area
        """
        serviceArea_polys = self.get_serviceArea_polygons(bus_radius, train_radius)
        # merge all polygons into one multipolygon
        serviceArea = serviceArea_polys.union_all()
        # convert polygon into a gdf
        serviceArea = gpd.GeoDataFrame({'geometry':[serviceArea]},crs=self.crs)

        if plot:
            
            # overlay bus and train nodes
            # get nearest nodes in graph G that is the closest to the locations of the bus stops and train exits
            bus_nodes = self.get_nearest_nodes(self.bus_stops_df)
            train_nodes = self.get_nearest_nodes(self.train_exits_df)
            # color the nodes according to isochrone then plot the street network
            node_colors = {}
            for node in bus_nodes:
                node_colors[node] = bus_c
            for node in train_nodes:
                node_colors[node] = train_c
            
            nc = [node_colors[node] if node in node_colors else "none" for node in self.G.nodes()]
            ns = [node_size if node in node_colors else 0 for node in self.G.nodes()]
            # plot the network then add isochrones as colored polygon patches
            fig, ax = ox.plot_graph(
                self.G, node_color=nc, node_size=ns,node_alpha=0.8,
                show=False, close=False, edge_color="#999999", edge_alpha=0.2
            )

            serviceArea.plot(ax=ax, color=serviceArea_c, ec="none", alpha=0.6, zorder=-1)
            plt.show()
        
        return serviceArea

    def calculate_serviceArea(self,serviceArea):
        """ 
        Args:
            serviceArea (gpd): polygon collection in a gdf that represents the union of the service area
        Returns:
            float: Area of service area in km2
        """
        serviceArea = serviceArea.to_crs({'proj':'cea'})
        area = serviceArea.geometry[0].area/10**6
        print(f'Area: {area} km2')
        return area
