# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import time
import networkx as nx
import joblib
import heapq
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px

def generate_routes(
    gdf,
    start,
    end,
    L_min,
    L_max,
    max_lens=[10, 25, 50, 100, 250],
    k_best=50
):
    gdf_sub = gdf[['u','v','length','score_totaal','wel_bankjes']]

    all_labels = []

    for ml in max_lens:

        res = pareto_paths_bankjes_gap(
            gdf_sub,
            start=start,
            end=end,
            L_min=L_min,
            L_max=L_max,
            epsilon=0.01,
            k_best=k_best,
            max_len=ml,
        )

        all_labels.extend(res)

    # unique routes
    unique = {}

    for l in all_labels:
        if len(l) < 7:
            continue
        route = tuple(l[6])
        if route not in unique:
            unique[route] = l

    labels_unique = list(unique.values())

    labels_unique = unique_by_score_gap(labels_unique)

    final_pareto = pareto_filter_bankjes_gap(labels_unique, epsilon=0)

    return final_pareto
def pareto_filter_dataframe(df, epsilon=0.0):
    '''
    Pareto-filter voor:
    - score: hoger is beter
    - max_gap: lager is beter
    '''

    keep = []

    for i, row_i in df.iterrows():

        dominated = False

        for j, row_j in df.iterrows():

            if i == j:
                continue

            # j domineert i als:
            # - j heeft hogere of gelijke score
            # - j heeft lagere of gelijke gap
            # - en strikt beter in minstens één

            if (
                row_j['score'] >= row_i['score'] * (1 - epsilon)
                and row_j['max_gap'] <= row_i['max_gap'] * (1 + epsilon)
                and (
                    row_j['score'] > row_i['score']
                    or row_j['max_gap'] < row_i['max_gap']
                )
            ):
                dominated = True
                break

        if not dominated:
            keep.append(i)

    return df.loc[keep].reset_index(drop=True)
def unique_by_score_gap(labels, digits=6):

    seen = set()
    out = []

    for l in labels:

        L = l[1]
        S = l[2]
        max_gap = l[5]

        if L <= 0:
            continue

        score_density = round(S / L, digits)
        gap = round(max_gap, digits)

        key = (score_density, gap)

        if key not in seen:
            seen.add(key)
            out.append(l)

    return out
def pareto_filter_bankjes_gap(labels, epsilon=0.0):

    pareto = []

    for l in labels:

        L, S, B, curr_gap, max_gap = l[1], l[2], l[3], l[4], l[5]

        if L <= 0:
            continue

        s = S / L
        g = max_gap

        dominated = False
        new_pareto = []

        for p in pareto:

            Lp, Sp, Bp, _, pg = p[1], p[2], p[3], p[4], p[5]

            sp = Sp / Lp
            gp = pg

            # p dominates l
            if (
                sp >= s * (1 - epsilon)
                and gp <= g * (1 + epsilon)
                and (sp > s or gp < g)
            ):
                dominated = True
                break

            # l dominates p
            if not (
                s >= sp * (1 - epsilon)
                and g <= gp * (1 + epsilon)
                and (s > sp or g < gp)
            ):
                new_pareto.append(p)

        if not dominated:
            new_pareto.append(l)
            pareto = new_pareto

    return pareto
def pareto_paths_bankjes_gap(
    gdf,
    start,
    end,
    L_min,
    L_max,
    epsilon=0.0,
    k_best=50,
    max_len=200,
    neighbor_limit=None
):

    # =========================
    # GRAPH
    # =========================

    adj = defaultdict(list)
    nodes = set()

    for r in gdf.itertuples(index=False):

        score_add = r.score_totaal * r.length
        bench = r.wel_bankjes

        adj[r.u].append((r.v, r.length, score_add, bench))

        nodes.add(r.u)
        nodes.add(r.v)

    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # =========================
    # INIT
    # =========================

    labels = defaultdict(list)
    pq = []

    start_mask = 1 << node_to_idx[start]

    start_label = (
        start,
        0.0,   # L
        0.0,   # S
        0.0,   # B
        0.0,   # curr_gap
        0.0,   # max_gap
        [start],
        start_mask
    )

    heapq.heappush(pq, (0.0, start_label))

    # =========================
    # MAIN LOOP
    # =========================

    while pq:

        if len(pq) > 50000:
            pq = heapq.nsmallest(25000, pq)

        _, (node, L, S, B, curr_gap, max_gap, path, visited) = heapq.heappop(pq)

        if len(path) > max_len:
            continue

        for nxt, length, s_add, bench in adj[node]:

            idx = node_to_idx[nxt]

            if visited & (1 << idx):
                continue

            L_new = L + length
            if L_new > L_max:
                continue

            S_new = S + s_add
            B_new = B + bench

            # =========================
            # GAP LOGIC (CRUCIAL)
            # =========================
            # gap = afstand sinds laatste bankje
            if bench == 1:
                curr_new = 0.0
            else:
                curr_new = curr_gap + length

            max_new = max(max_gap, curr_new)

            # =========================
            # LABEL UPDATE
            # =========================

            labels_nxt = labels[nxt]

            s_new = S_new / L_new

            dominated = False
            keep = []

            for old in labels_nxt:

                Lo, So, Bo, _, old_gap = old[1], old[2], old[3], old[4], old[5]

                s_old = So / Lo

                if (
                    s_old >= s_new * (1 - epsilon)
                    and old_gap <= max_new * (1 + epsilon)
                    and (s_old > s_new or old_gap < max_new)
                ):
                    dominated = True
                    break

                if not (
                    s_new >= s_old * (1 - epsilon)
                    and max_new <= old_gap * (1 + epsilon)
                    and (s_new > s_old or max_new < old_gap)
                ):
                    keep.append(old)

            if dominated:
                continue

            labels_nxt[:] = keep

            new_mask = visited | (1 << idx)

            new_label = (
                nxt,
                L_new,
                S_new,
                B_new,
                curr_new,
                max_new,
                path + [nxt],
                new_mask
            )

            labels_nxt.append(new_label)

            # =========================
            # K BEST
            # =========================

            if len(labels_nxt) > k_best:
                labels_nxt.sort(key=lambda l: (-(l[2] / l[1]), l[5]))
                labels_nxt[:] = labels_nxt[:k_best]

            # =========================
            # PRIORITY (BALANCED)
            # =========================
            priority = -(S_new / L_new) + (max_new / L_max)

            heapq.heappush(pq, (priority, new_label))

    # =========================
    # OUTPUT
    # =========================

    return [
        l for l in labels[end]
        if l[1] >= L_min
    ]
def pareto_path_to_gdf(gdf, path):

    df_route = pd.DataFrame({'u': path})
    df_route['v'] = df_route['u'].shift(-1)
    df_route = df_route.dropna()

    df_route = gdf.merge(df_route, on=['u', 'v'])

    return df_route
def select_best_pareto_route(pareto, L_min, L_max, max_gap):

    feasible = []

    for r in pareto:
        L = r[1]
        S = r[2]
        gap = r[5]

        if L_min <= L <= L_max and gap <= max_gap:
            score_density = S / L if L > 0 else 0
            feasible.append((score_density, L, S, gap, r))


    if len(feasible) > 0:

        best = max(
            feasible,
            key=lambda x: (x[0], -abs(L_max - x[1]))
        )

        _, L, S, _, route = best
        return route, L, S


    fallback = min(pareto, key=lambda r: r[5])

    L = fallback[1]
    S = fallback[2]

    return fallback, L, S
def compute_route_score_from_gdf(gdf, route):
    total_length = 0
    weighted_score = 0

    for i in range(len(route) - 1):

        u = route[i]
        v = route[i + 1]

        edge = gdf[((gdf.u == u) & (gdf.v == v)) |
                   ((gdf.u == v) & (gdf.v == u))]

        if len(edge) == 0:
            continue

        length = edge['length'].values[0]
        score = edge['Score'].values[0]  # jouw bestaande kolom

        total_length += length
        weighted_score += score * length

    if total_length == 0:
        return 0

    return weighted_score / total_length
def evaluate_route_from_gdf_bankjes(gdf, route):
    """
    Berekent:
    - distance
    - lengte-gewogen score (omgevingsscore)
    - max gap zonder bankjes
    (Streamlit-safe, no matrix build)
    """

    # =========================
    # EDGE LOOKUP TABLE
    # =========================
    df = gdf[['u', 'v', 'length', 'score_totaal', 'wel_bankjes']].copy()

    # maak symmetric (belangrijk!)
    df_rev = df.rename(columns={'u': 'v', 'v': 'u'})
    df_all = pd.concat([df, df_rev], ignore_index=True)

    lookup = df_all.set_index(['u', 'v'])

    distance = 0.0
    weighted_score = 0.0

    max_gap = 0.0
    current_gap = 0.0

    # =========================
    # ROUTE LOOP
    # =========================
    for i in range(len(route) - 1):

        u = route[i]
        v = route[i + 1]

        try:
            edge = lookup.loc[(u, v)]
        except KeyError:
            # silently skip instead of crash
            continue

        # als duplicate rows bestaan
        if isinstance(edge, pd.DataFrame):
            edge = edge.iloc[0]

        length = float(edge['length'])
        score_s = float(edge['score_totaal'])
        bench = float(edge['wel_bankjes'])

        distance += length
        weighted_score += score_s * length

        # =========================
        # GAP LOGIC
        # =========================
        if bench == 1:
            max_gap = max(max_gap, current_gap)
            current_gap = 0.0
        else:
            current_gap += length

    max_gap = max(max_gap, current_gap)

    # =========================
    # NORMALIZED SCORE
    # =========================
    score = (
        weighted_score / distance
        if distance > 0 else 0
    )

    return distance, score, max_gap

def evaluate_route_score_only(gdf, route):
    df = gdf[['u', 'v', 'length', 'score_totaal']].copy()

    df_rev = df.rename(columns={'u': 'v', 'v': 'u'})
    lookup = pd.concat([df, df_rev]).set_index(['u', 'v'])

    total_score = 0.0
    total_length = 0.0

    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]

        try:
            edge = lookup.loc[(u, v)]
        except KeyError:
            continue

        if isinstance(edge, pd.DataFrame):
            edge = edge.iloc[0]

        length = float(edge['length'])
        score = float(edge['score_totaal'])

        total_score += score * length
        total_length += length

    return total_score / total_length if total_length > 0 else 0

def pareto_to_df(pareto):

    rows = []

    for i, p in enumerate(pareto):

        try:
            L = p[1]
            S = p[2]
            gap = p[5]

            if L > 0:

                rows.append({
                    "id": i,
                    "max_gap": gap,
                    "gemiddelde_score": S / L,
					"afstand": L,   
                })

        except Exception:
            pass

    return pd.DataFrame(rows)


def plot_pareto(df, selected_gap=None, selected_score=None, selected_distance=None):

    fig = go.Figure()

    # Alle punten
    fig.add_trace(go.Scatter(
        x=df["max_gap"],
        y=df["gemiddelde_score"],
        mode="markers",
        name="Andere mogelijke routes",
        marker=dict(size=8, opacity=0.7),
        customdata=df["afstand"],

        hovertemplate=
            "Max gap: %{x:.2f} m<br>" +
            "Score: %{y:.2f}<br>" +
            "Afstand: %{customdata:.2f} m" +
            "<extra></extra>"
    ))

    # Geselecteerd punt
    if selected_gap is not None and selected_score is not None:
        fig.add_trace(go.Scatter(
            x=[selected_gap],
            y=[selected_score],
            mode="markers",
            name="Geselecteerde route",
            marker=dict(size=10, color="red"),

            hovertemplate=
                "Max gap: %{x:.2f} m<br>" +
                "Score: %{y:.2f}<br>" +
                (f"Afstand: {selected_distance:.2f} m<br>" if selected_distance is not None else "") +
                "<extra></extra>"
        ))

    fig.update_layout(
        title="Alle mogelijke routes met bijbehorende omgevingsscores en maximale gap",
        xaxis_title="Maximale afstand tussen bankjes (m)",
        yaxis_title="Gemiddelde omgevingsscore",
        template="plotly_white",
        hovermode="closest"
    )

    return fig

###LUNCHROUTES
def labels_to_dataframe(pareto):

    rows = []

    for i,l in enumerate(pareto):

        L = l[1]
        S = l[2]

        rows.append({

            "id": i,
            "afstand": L,
            "gemiddelde_score": S/L,
            "route": l[3]

        })

    return pd.DataFrame(rows)


def pareto_paths_2d_ultrafast(
    gdf,
    start,
    end,
    L_min,
    L_max,
    epsilon=0.01,
    k_best=50,
    max_len=50,
    neighbor_limit=7
):

    # =====================================================
    # adjacency
    # =====================================================

    adj = defaultdict(list)
    nodes = set()

    for r in gdf.itertuples(index=False):

        adj[r.u].append(
            (
                r.v,
                r.length,
                r.score_totaal
            )
        )

        nodes.add(r.u)
        nodes.add(r.v)

    # =====================================================
    # node index (bitmask)
    # =====================================================

    node_list = list(nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    # =====================================================
    # Dijkstra pruning
    # =====================================================

    def dijkstra(adj, start):

        dist = {start: 0.0}
        pq = [(0.0, start)]

        while pq:

            d, u = heapq.heappop(pq)

            if d > dist[u]:
                continue

            for v, length, _ in adj[u]:

                nd = d + length

                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))

        return dist

    def build_reverse_adj(gdf):

        rev = defaultdict(list)

        for r in gdf.itertuples(index=False):
            rev[r.v].append(
                (
                    r.u,
                    r.length,
                    r.score_totaal
                )
            )

        return rev

    d_start = dijkstra(adj, start)
    d_end = dijkstra(build_reverse_adj(gdf), end)

    valid_nodes = {
        v for v in d_start
        if v in d_end and (d_start[v] + d_end[v] <= L_max)
    }

    valid_nodes.add(end)

    # =====================================================
    # labels
    # =====================================================

    labels = defaultdict(list)

    pq = []

    start_mask = 1 << node_to_idx[start]

    # label:
    # (node, length, score, path, visited)

    start_label = (
        start,
        0.0,
        0.0,
        [start],
        start_mask
    )

    heapq.heappush(pq, (0.0, start_label))

    # =====================================================
    # main loop
    # =====================================================

    while pq:

        # memory protection
        if len(pq) > 20000:
            pq = heapq.nsmallest(10000, pq)

        _, (
            node,
            L,
            S,
            path,
            visited
        ) = heapq.heappop(pq)

        if len(path) > max_len:
            continue

        neighbors = adj[node]

        # =================================================
        # neighbor pruning
        # =================================================

        if len(neighbors) > neighbor_limit:

            neighbors = sorted(
                neighbors,
                key=lambda x: x[2] / (x[1] + 1e-6),
                reverse=True
            )[:neighbor_limit]

        # =================================================
        # expand neighbors
        # =================================================

        for nxt, length, s_val in neighbors:

            if nxt not in valid_nodes:
                continue

            idx = node_to_idx[nxt]

            # avoid cycles
            if visited & (1 << idx):
                continue

            remaining = d_end.get(nxt, float("inf"))

            # impossible to reach target
            if L + length + remaining > L_max:
                continue

            L_new = L + length

            if L_new > L_max:
                continue

            # additive total score
            S_new = S + s_val * length

            avg_new = S_new / L_new

            labels_nxt = labels[nxt]

            dominated = False
            new_list = []

            # =============================================
            # Pareto dominance:
            #
            # minimize distance
            # maximize average score
            # =============================================

            for old in labels_nxt:

                L_old, S_old = old[1], old[2]

                avg_old = S_old / L_old

                # old dominates new
                if (
                    L_old <= L_new * (1 + epsilon) and
                    avg_old >= avg_new * (1 - epsilon)
                ):
                    dominated = True
                    break

                # keep old if new does NOT dominate old
                if not (
                    L_new <= L_old * (1 + epsilon) and
                    avg_new >= avg_old * (1 - epsilon)
                ):
                    new_list.append(old)

            if dominated:
                continue

            labels_nxt[:] = new_list

            # =============================================
            # create new label
            # =============================================

            new_mask = visited | (1 << idx)

            new_label = (
                nxt,
                L_new,
                S_new,
                path + [nxt],
                new_mask
            )

            labels_nxt.append(new_label)

            # =============================================
            # keep only k best
            # =============================================

            if len(labels_nxt) > k_best:

                labels_nxt.sort(
                    key=lambda l: (
                        l[1],                  # shorter first
                        -(l[2] / l[1])        # higher avg score first
                    )
                )

                labels_nxt[:] = labels_nxt[:k_best]

            # =============================================
            # queue priority
            # =============================================

            priority = -(avg_new)

            heapq.heappush(
                pq,
                (priority, new_label)
            )

    # =====================================================
    # collect results
    # =====================================================

    end_labels = [
        l for l in labels[end]
        if l[1] >= L_min
    ]

    return end_labels


# =========================================================
# Unique filter
# =========================================================

def unique_by_density(labels, tol=1e-6):

    seen = set()
    out = []

    for l in labels:

        L, S = l[1], l[2]

        if L == 0:
            continue

        avg = round(S / L, 6)
        dist = round(L, 1)

        key = (dist, avg)

        if key not in seen:
            seen.add(key)
            out.append(l)

    return out


# =========================================================
# Pareto filter
# =========================================================

def pareto_filter(labels, epsilon=0.0):

    pareto = []

    for l in labels:

        L, S = l[1], l[2]

        if L == 0:
            continue

        avg = S / L

        dominated = False
        new_list = []

        for p in pareto:

            Lp, Sp = p[1], p[2]

            avg_p = Sp / Lp

            # p dominates l
            if (
                Lp <= L * (1 + epsilon) and
                avg_p >= avg * (1 - epsilon)
            ):
                dominated = True
                break

            # l dominates p
            if not (
                L <= Lp * (1 + epsilon) and
                avg >= avg_p * (1 - epsilon)
            ):
                new_list.append(p)

        if not dominated:
            new_list.append(l)
            pareto = new_list

    return pareto
def generate_pareto_routes_2d(
    gdf,
    start,
    end,
    L_min,
    L_max,
    max_lens=[10,25,50,100,250]
):

    all_labels = []

    for ml in max_lens:

        res = pareto_paths_2d_ultrafast(
            gdf=gdf,
            start=start,
            end=end,
            L_min=L_min,
            L_max=L_max,
            epsilon=0.01,
            k_best=50,
            max_len=ml,
            neighbor_limit=7
        )

        all_labels.extend(res)

    unique = {}

    for l in all_labels:

        h = l[4]

        if h not in unique:
            unique[h] = l

    labels_unique = list(unique.values())

    labels_unique_filtered = unique_by_density(
        labels_unique
    )

    final_pareto = pareto_filter(
        labels_unique_filtered
    )

    return final_pareto

def plot_pareto_2d(
    df,
    selected_distance=None,
    selected_score=None
):

    fig = go.Figure()
    
    # Alle routes
    fig.add_trace(
        go.Scatter(
            x=df["afstand"],
            y=df["gemiddelde_score"],
            mode="markers",
            name="Andere mogelijke routes",
            marker=dict(size=8, opacity=0.7),
            customdata=df["id"],
            hovertemplate=
            "Route ID: %{customdata}<br>" +
            "Afstand: %{x:.2f} m<br>" +
            "Gemiddelde score: %{y:.2f}" +
            "<extra></extra>"
                )
            )

    # Geselecteerde route
    if (
        selected_distance is not None
        and selected_score is not None
    ):

        fig.add_trace(
            go.Scatter(
                x=[selected_distance],
                y=[selected_score],
                mode="markers",
                name="Geselecteerde route",
                marker=dict(
                    size=10,
                    color="red"
                ),

                hovertemplate=
                    "Afstand: %{x:.2f} m<br>" +
                    "Gemiddelde score: %{y:.2f}" +
                    "<extra></extra>"
            )
        )

    fig.update_layout(
        title="Alle mogelijke ideale lunchroutes",
        xaxis_title="Afstand (m)",
        yaxis_title="Gemiddelde omgevingsscore",
        template="plotly_white",
        hovermode="closest"
    )

    return fig


