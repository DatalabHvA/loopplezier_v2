import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.sparse.csgraph import shortest_path
import random

def small_matrices(a, s, g_max, start, end):
    G = csr_matrix(a)

    dist_start = shortest_path(G, directed=False, indices=start)
    dist_end   = shortest_path(G, directed=False, indices=end)

    reachable = np.intersect1d(
        np.where(dist_start < g_max)[0],
        np.where(dist_end < g_max)[0]
    )

    a_new = a[np.ix_(reachable, reachable)]
    s_new = s[np.ix_(reachable, reachable)]

    a_new[a_new >= 100000] = np.inf
    s_new[a_new == np.inf] = 0

    start_new = int(np.where(reachable == start)[0][0])
    end_new   = int(np.where(reachable == end)[0][0])

    return a_new, s_new, start_new, end_new, reachable

def preprocess(gdf, start, end, g_max, mandatory_nodes):
    a = gdf.pivot(index='u', columns='v', values='length').fillna(100000).values
    s = gdf.pivot(index='u', columns='v', values='score_totaal').fillna(0).values

    s = s * a

    a, s, start, end, indices = small_matrices(a, s, g_max, start, end)

    # voeg missende mandatory nodes toe
    missing = [n for n in mandatory_nodes if n not in indices]
    if missing:
        indices = np.unique(np.concatenate([indices, missing]))
        a = a[np.ix_(indices, indices)]
        s = s[np.ix_(indices, indices)]

        a[a >= 100000] = np.inf
        s[a == np.inf] = 0

        start = int(np.where(indices == start)[0][0])
        end   = int(np.where(indices == end)[0][0])

    return a, s, start, end, indices

def build_df_route_from_grasp(gdf, best_route):
    df_route = pd.DataFrame({'u': best_route})
    df_route['v'] = df_route['u'].shift(-1)
    df_route = df_route.dropna()

    # 🔧 FIX: forceer int types (cruciaal voor merge!)
    df_route['u'] = df_route['u'].astype(int)
    df_route['v'] = df_route['v'].astype(int)

    gdf['u'] = gdf['u'].astype(int)
    gdf['v'] = gdf['v'].astype(int)

    df_route = df_route.merge(gdf, on=['u','v'], how='left')

    return df_route
def build_graph(a, s):

    G = nx.Graph()

    n = a.shape[0]

    eps = 1e-6

    # verzamel quality values
    qualities = []

    for i in range(n):
        for j in range(i + 1, n):

            if np.isfinite(a[i, j]) and a[i, j] < 100000:

                length = a[i, j]

                quality = s[i, j] / (length + eps)

                qualities.append(quality)

    # normalisatie
    q_min = min(qualities)
    q_max = max(qualities)

    for i in range(n):
        for j in range(i + 1, n):

            if np.isfinite(a[i, j]) and a[i, j] < 100000:

                length = a[i, j]

                quality = s[i, j] / (length + eps)

                # schaal naar [0,1]
                quality_norm = (
                    (quality - q_min)
                    / (q_max - q_min + eps)
                )

                # altijd positief
                # hoge quality => lage cost
                cost = length * ((1+1e-6) - quality_norm)

                G.add_edge(
                    i,
                    j,
                    length=length,
                    score=s[i, j],
                    quality=quality,
                    quality_norm=quality_norm,
                    cost=cost
                )

    return G
def shortest_path_info(G, i, j, visited):

    # behoud visited constraint
    allowed = set(G.nodes()) - visited

    allowed.add(i)
    allowed.add(j)

    H = G.subgraph(allowed)

    try:


        path = nx.shortest_path(
            H,
            i,
            j,
            weight='cost'
        )

    except nx.NetworkXNoPath:
        return None

    d = 0
    sc = 0

    for u, v in zip(path[:-1], path[1:]):

        edge = G[u][v]

        d += edge['length']
        sc += edge['score']
    
    return path, d, sc
def grasp_route(
    G,
    start,
    end,
    mandatory,
    g_max,
    alpha=0.3
):

    remaining = set(mandatory)

    current = start

    route = [start]

    visited = {start}

    total_length = 0
    total_score = 0

    while remaining:

        candidates = []

        for node in remaining:

            res = shortest_path_info(
                G,
                current,
                node,
                visited
            )

            if res is None:
                continue

            path, d, sc = res

            # utility
            value = sc / (d +1e-6)

            candidates.append(
                (node, path, d, sc, value)
            )

        if not candidates:
            return None

        # hoogste value eerst
        candidates.sort(
            key=lambda x: x[4],
            reverse=True
        )

        max_v = candidates[0][4]
        min_v = candidates[-1][4]

        threshold = max_v - alpha * (max_v - min_v)

        rcl = [
            c for c in candidates
            if c[4] >= threshold
        ]

        chosen = random.choice(rcl)

        node, path, d, sc, value = chosen

        if total_length + d > g_max:
            continue

        # voeg path toe
        route.extend(path[1:])

        # visited bijhouden
        visited.update(path[1:])

        total_length += d
        total_score += sc

        current = node

        remaining.remove(node)

    # eindnode toevoegen
    res = shortest_path_info(
        G,
        current,
        end,
        visited
    )

    if res is None:
        return None

    path, d, sc = res

    if total_length + d > g_max:
        return None

    route.extend(path[1:])

    total_length += d
    total_score += sc
    if not candidates:
        print("FAIL STATE")
        print("current:", current)
        print("remaining:", remaining)
        print("budget left:", g_max - total_length)
        return None
    return route, total_length, total_score
def run_grasp(
    a,
    s,
    start,
    end,
    mandatory,
    g_max,
    n_iter=100,
    alpha=0.5
):

    G = build_graph(a, s)

    best_route = None

    best_value = -np.inf

    for _ in range(n_iter):

        result = grasp_route(
            G,
            start,
            end,
            mandatory,
            g_max,
            alpha
        )

        if result is None:
            continue

        route, d, sc = result

        # globale objective
        value = sc / (d + 1e-6)

        if value > best_value:

            best_value = value
            best_route = route

    return best_route, best_value


def calculate_mandatory_route(
    gdf,
    start,
    end,
    mandatory_nodes,
    max_dist
):

    a_final, s_final, start_final, end_final, indices = preprocess(
        gdf,
        start=start,
        end=end,
        g_max=max_dist,
        mandatory_nodes=mandatory_nodes
    )

    mandatory_local = [
        int(np.where(indices == n)[0][0])
        for n in mandatory_nodes
    ]

    best_route, best_value = run_grasp(
        a=a_final,
        s=s_final,
        start=start_final,
        end=end_final,
        mandatory=mandatory_local,
        g_max=max_dist,
        n_iter=50,
        alpha=0.5
    )

    if best_route is None:
        return None, None, None

    best_route_original = indices[best_route]

    df_route = build_df_route_from_grasp(
        gdf,
        best_route_original
    )

    distance = df_route["length"].sum()

    score = (
        (df_route["score_totaal"] * df_route["length"]).sum()
        /
        distance
    )

    return (
        df_route,
        distance,
        score
    )