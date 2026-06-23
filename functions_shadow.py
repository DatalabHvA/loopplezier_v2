import heapq
from collections import defaultdict
import pandas as pd
import plotly.graph_objects as go

###SCHADUWBEDEKKING
def pareto_path_to_gdf(gdf, path):

    df_route = pd.DataFrame({'u': path})
    df_route['v'] = df_route['u'].shift(-1)
    df_route = df_route.dropna()

    df_route = gdf.merge(df_route, on=['u', 'v'])

    return df_route
def unique_by_density(labels, digits=6):

    seen = set()
    out = []

    for l in labels:

        L, S, SH = l[1], l[2], l[3]

        if L <= 0:
            continue

        s = round(S / L, digits)
        sh = round(SH / L, digits)

        if (s, sh) not in seen:
            seen.add((s, sh))
            out.append(l)

    return out

def pareto_filter(labels, epsilon=0.0):

    pareto = []

    for l in labels:

        L, S, SH = l[1], l[2], l[3]

        if L <= 0:
            continue

        s = S / L
        sh = SH / L

        dominated = False
        new_pareto = []

        for p in pareto:

            Lp, Sp, SHp = p[1], p[2], p[3]

            sp = Sp / Lp
            shp = SHp / Lp

            # p domineert l
            if (
                sp >= s * (1 - epsilon)
                and
                shp >= sh * (1 - epsilon)
                and
                (
                    sp > s
                    or shp > sh
                )
            ):
                dominated = True
                break

            # l domineert p
            if not (
                s >= sp * (1 - epsilon)
                and
                sh >= shp * (1 - epsilon)
                and
                (
                    s > sp
                    or sh > shp
                )
            ):
                new_pareto.append(p)

        if not dominated:
            new_pareto.append(l)
            pareto = new_pareto

    return pareto
def pareto_paths(
    gdf,
    start,
    end,
    L_min,
    L_max,
    epsilon=0.0,
    k_best=100,
    max_len=200,
    neighbor_limit=None
):

    # =====================================================
    # BUILD GRAPH
    # =====================================================

    adj = defaultdict(list)
    nodes = set()
    for r in gdf.itertuples(index=False):


        s_add = r.score_totaal * r.length
        sh_add = r.score_schaduw * r.length

        adj[r.u].append(
            (
                r.v,
                r.length,
                s_add,
                sh_add
            )
        )

        nodes.add(r.u)
        nodes.add(r.v)

    # =====================================================
    # NODE INDEX
    # =====================================================

    node_list = list(nodes)
    node_to_idx = {
        n: i
        for i, n in enumerate(node_list)
    }

    # =====================================================
    # DIJKSTRA
    # =====================================================

    def dijkstra(adj, start):

        dist = {start: 0.0}

        pq = [(0.0, start)]

        while pq:

            d, u = heapq.heappop(pq)

            if d > dist[u]:
                continue

            for v, length, _, _ in adj[u]:

                nd = d + length

                if v not in dist or nd < dist[v]:

                    dist[v] = nd

                    heapq.heappush(
                        pq,
                        (nd, v)
                    )

        return dist

    def reverse_adj(gdf):

        rev = defaultdict(list)

        for r in gdf.itertuples(index=False):

            rev[r.v].append(
                (
                    r.u,
                    r.length,
                    0,
                    0
                )
            )

        return rev

    d_start = dijkstra(adj, start)
    d_end = dijkstra(reverse_adj(gdf), end)

    valid_nodes = {
        v for v in d_start
        if (
            v in d_end
            and
            d_start[v] + d_end[v] <= L_max
        )
    }

    valid_nodes.add(end)

    # =====================================================
    # LABELS
    # =====================================================

    labels = defaultdict(list)

    pq = []

    start_mask = 1 << node_to_idx[start]

    start_label = (
        start,      # node
        0.0,        # length
        0.0,        # additive score
        0.0,        # additive shadow
        [start],    # path
        start_mask
    )

    heapq.heappush(
        pq,
        (0.0, start_label)
    )

    # =====================================================
    # MAIN LOOP
    # =====================================================

    while pq:

        # if len(pq) > 50000:
        #     pq = heapq.nsmallest(25000, pq)

        _, (
            node,
            L,
            S,
            SH,
            path,
            visited
        ) = heapq.heappop(pq)

        if len(path) > max_len:
            continue

        neighbors = adj[node]

        # =================================================
        # OPTIONAL HEURISTIC PRUNING
        # =================================================

        if (
            neighbor_limit is not None
            and
            len(neighbors) > neighbor_limit
        ):

            neighbors = sorted(
                neighbors,
                key=lambda x:
                    (
                        (x[2] / x[1]) +
                        (x[3] / x[1])
                    ),
                reverse=True
            )[:neighbor_limit]

        # =================================================
        # EXPAND
        # =================================================

        for nxt, length, s_add, sh_add in neighbors:

            if nxt not in valid_nodes:
                continue

            idx = node_to_idx[nxt]

            # simpele paden
            if visited & (1 << idx):
                continue

            remaining = d_end.get(
                nxt,
                L_max + 1
            )

            if L + length + remaining > L_max:
                continue

            L_new = L + length

            if L_new > L_max:
                continue

            # =============================================
            # EXACT CONSISTENT MET ACO
            # =============================================

            S_new = S + s_add
            SH_new = SH + sh_add

            if L_new <= 0:
                continue

            score_density = S_new / L_new
            shadow_density = SH_new / L_new

            labels_nxt = labels[nxt]

            dominated = False
            keep = []

            # =============================================
            # PARETO DOMINANCE
            # =============================================

            for old in labels_nxt:

                Lo, So, SHo = old[1], old[2], old[3]

                s_old = So / Lo
                sh_old = SHo / Lo

                # old domineert new
                if (
                    s_old >= score_density * (1 - epsilon)
                    and
                    sh_old >= shadow_density * (1 - epsilon)
                    and
                    (
                        s_old > score_density
                        or
                        sh_old > shadow_density
                    )
                ):
                    dominated = True
                    break

                # new domineert old
                if not (
                    score_density >= s_old * (1 - epsilon)
                    and
                    shadow_density >= sh_old * (1 - epsilon)
                    and
                    (
                        score_density > s_old
                        or
                        shadow_density > sh_old
                    )
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
                SH_new,
                path + [nxt],
                new_mask
            )

            labels_nxt.append(new_label)

            # =============================================
            # K BEST
            # =============================================
            if (
                k_best is not None
                and
                len(labels_nxt) > k_best
            ):

                labels_nxt.sort(
                    key=lambda l:
                       -(l[2] + 3*l[3])
                )

                labels_nxt[:] = labels_nxt[:k_best]

            # =============================================
            # PRIORITY
            # =============================================
            priority =  - S_new -  SH_new
            heapq.heappush(
                pq,
                (
                    priority,
                    new_label
                )
            )

    # =====================================================
    # FINAL
    # =====================================================

    end_labels = [
        l for l in labels[end]
        if l[1] >= L_min
    ]

    return end_labels
def generate_pareto_routes_2d(gdf, start, end, L_min, L_max, max_lens):
    all_labels = []

    for ml in max_lens:

        res = pareto_paths(
            gdf,
            start=start,
            end=end,
            L_min=L_min,
            L_max=L_max,
            max_len=ml
        )

        all_labels.extend(res)

    # unique paths
    unique = {}
    for l in all_labels:
        h = tuple(l[4])
        if h not in unique:
            unique[h] = l

    labels_unique = list(unique.values())
    labels_unique = unique_by_density(labels_unique)
    labels_unique = pareto_filter(labels_unique)

    return labels_unique

def pareto_to_df_2d(labels):
    rows = []

    for l in labels:

        node, L, S, SH, path, visited = l

        if L <= 0:
            continue

        rows.append({
            "afstand": float(L),
            "score_density": float(S / L),
            "shadow_density": float(SH / L),
            "score_total": float(S),
            "shadow_total": float(SH),
            "path": path
        })

    return pd.DataFrame(rows)

def plot_pareto(df, selected_shadow=None, selected_score=None, selected_distance=None):

    fig = go.Figure()

    # Alle oplossingen
    fig.add_trace(go.Scatter(
        x=df["shadow_density"] * 100,   # percentage
        y=df["score_density"],
        mode="markers",
        name="Andere mogelijke routes",
        marker=dict(size=8, opacity=0.7),
        customdata=df[["afstand"]],

        hovertemplate=
            "Schaduw: %{x:.2f}%<br>" +
            "Score: %{y:.2f}<br>" +
            "Afstand: %{customdata[0]:.0f} m" +
            "<extra></extra>"
    ))

    # Geselecteerde route
    if selected_shadow is not None and selected_score is not None:

        fig.add_trace(go.Scatter(
            x=[selected_shadow],
            y=[selected_score],
            mode="markers",
            name="Geselecteerde route",
            marker=dict(size=10, color="red"),

            hovertemplate=
                "Schaduw: %{x:.2f}%<br>" +
                "Score: %{y:.2f}<br>" +
                (f"Afstand: {selected_distance:.0f} m<br>" if selected_distance is not None else "") +
                "<extra></extra>"
        ))

    fig.update_layout(
        title="Alternatieve routes met schaduw vs omgevingsscore",
        xaxis_title="Schaduwbedekking (%)",
        yaxis_title="Gemiddelde omgevingsscore",
        template="plotly_white",
        hovermode="closest"
    )

    return fig