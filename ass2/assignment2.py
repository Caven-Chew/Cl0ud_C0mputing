"""
SC4052 Cloud Computing — Assignment 2, Question 6
==================================================
PageRank: Theory, Implementation, and GPTBot-Like AI Crawler Prioritisation

Sections
────────
  Part A — Core PageRank (web-Google.txt)
    1.  Graph loading
    2.  Column-stochastic matrix construction (CSC sparse)
    3.  Toy graph: per-node p-sweep illustration  (Figure 1)
    4.  Solver 1 — Power iteration
    5.  Solver 2 — Closed-form (sparse direct solve)
    6.  Solver 3 — Monte Carlo random-walk simulator
    7.  Three-way numerical comparison              (Figure 2)
    8.  Full graph evaluation on web-Google.txt
    9.  Effect of p on distribution                (Figure 4)
    10. Log-log rank distribution                  (Figure 3)

  Part B — Extension (i): GPTBot-Like Crawler
    11. Demo URL web graph (dict input as per spec)
    12. PageRank computation on URL graph
    13. Permissive-Authority Heuristic (ONE heuristic)
    14. get_top_k_urls() — main crawler function
    15. Crawler results & figure                   (Figure 5)

Usage
─────
    python assignment2.py

Outputs (figures saved to working directory)
────────────────────────────────────────────
    toy_psweep.png          Figure 1  — toy graph p-sweep
    method_comparison.png   Figure 2  — three-way solver validation
    rank_dist.png           Figure 3  — power-law score distribution
    p_effect.png            Figure 4  — Gini / entropy / convergence vs p
    crawler_ext.png         Figure 5  — GPTBot crawler priority scores

Requirements
────────────
    pip install numpy scipy matplotlib
    Dataset: web-Google.txt in the same directory
             (https://snap.stanford.edu/data/web-Google.html)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from collections import defaultdict
import time
import re
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ═══════════════════════════════════════════════════════════════
# PART A — CORE PAGERANK
# ═══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# 1.  Graph Loading
# ─────────────────────────────────────────────────────────────

def load_graph(filepath):
    """
    Stream-parse an edge-list file.
    Skips comment lines (#) and self-loops.

    Returns
    -------
    out_adj : dict  {node_id: [neighbour_ids, ...]}
    nodes   : sorted list of all node IDs
    n_edges : number of directed edges loaded
    """
    out_adj = defaultdict(list)
    nodes   = set()
    n_edges = 0
    with open(filepath) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:          # skip self-loops
                continue
            out_adj[u].append(v)
            nodes.update((u, v))
            n_edges += 1
    nodes = sorted(nodes)
    return out_adj, nodes, n_edges


# ─────────────────────────────────────────────────────────────
# 2.  Column-Stochastic Sparse Matrix
# ─────────────────────────────────────────────────────────────

def build_stochastic(out_adj, nodes):
    """
    Build the column-stochastic hyperlink matrix S in CSC format.
    Dangling nodes (no valid out-edges) are tracked via a boolean
    mask; their rank mass is redistributed uniformly in the solver.

    Returns
    -------
    A            : scipy CSC sparse matrix (n × n)
    node_to_idx  : dict {node_id: column/row index}
    dangling_mask: boolean array, True where node is dangling
    """
    n   = len(nodes)
    nti = {nd: i for i, nd in enumerate(nodes)}
    rows, cols, data = [], [], []
    dangling = []

    for nd in nodes:
        idx  = nti[nd]
        outs = [nti[v] for v in out_adj.get(nd, []) if v in nti]
        if not outs:
            dangling.append(idx)
            continue
        w = 1.0 / len(outs)
        for dst in outs:
            rows.append(dst)
            cols.append(idx)
            data.append(w)

    A = sp.csc_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    dangling_mask = np.zeros(n, dtype=bool)
    dangling_mask[dangling] = True
    return A, nti, dangling_mask


# ─────────────────────────────────────────────────────────────
# 3.  Toy Graph: p-sweep illustration
# ─────────────────────────────────────────────────────────────

def build_toy_graph():
    """
    Six-node directed graph with three distinct authority levels:
      • Main cycle  : 0 → 1 → 2 → 0
      • Sink pair   : 3 → 4 → 3
      • Bridge node : 5 → 0  (makes graph strongly connected)
    Additional edges: 0 → 2, 1 → 3, 3 → 5
    """
    out_adj = {
        0: [1, 2],
        1: [2, 3],
        2: [0],
        3: [4, 5],
        4: [3],
        5: [0],
    }
    return out_adj, sorted(out_adj.keys())


def _pagerank_dense(A_dense, p=0.15, tol=1e-12, max_iter=1000):
    """Power iteration on a small dense matrix (toy graph only)."""
    n = A_dense.shape[0]
    r = np.ones(n) / n
    e = np.ones(n) / n
    for _ in range(max_iter):
        r_new = (1 - p) * A_dense @ r + p * e
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


def toy_psweep(out_adj, nodes):
    """Compute PageRank for each node across p ∈ [0.01, 0.99]."""
    n   = len(nodes)
    nti = {nd: i for i, nd in enumerate(nodes)}
    A   = np.zeros((n, n))
    for nd in nodes:
        outs = [nti[v] for v in out_adj.get(nd, []) if v in nti]
        if outs:
            for dst in outs:
                A[dst, nti[nd]] = 1.0 / len(outs)
        else:
            A[:, nti[nd]] = 1.0 / n

    p_vals = np.linspace(0.01, 0.99, 120)
    scores = np.zeros((n, len(p_vals)))
    for j, pv in enumerate(p_vals):
        scores[:, j] = _pagerank_dense(A, p=pv)
    return p_vals, scores


def plot_toy_psweep(p_vals, scores, nodes, output='toy_psweep.png'):
    n      = len(nodes)
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    spread = scores.max(axis=0) - scores.min(axis=0)
    idx_015 = np.argmin(np.abs(p_vals - 0.15))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for i in range(n):
        axes[0].plot(p_vals, scores[i], color=colors[i],
                     label=f'Node {nodes[i]}', linewidth=1.8)
    axes[0].axvline(0.15, color='gray', linestyle='--', linewidth=1,
                    label='p=0.15 (standard)')
    axes[0].set_xlabel('Teleportation probability p')
    axes[0].set_ylabel('PageRank score')
    axes[0].set_title('Per-Node PageRank vs p\n(6-node toy graph)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(p_vals, 0, spread, alpha=0.4, color='steelblue')
    axes[1].plot(p_vals, spread, color='steelblue', linewidth=2)
    axes[1].axvline(0.15, color='gray', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Teleportation probability p')
    axes[1].set_ylabel('Score spread (max − min)')
    axes[1].set_title('Ranking Inequality vs p\n(spread collapses as p→1)')
    axes[1].grid(True, alpha=0.3)

    bar_scores = scores[:, idx_015]
    bars = axes[2].bar([f'Node {nd}' for nd in nodes], bar_scores,
                       color=colors, edgecolor='white')
    axes[2].set_xlabel('Node')
    axes[2].set_ylabel('PageRank score')
    axes[2].set_title('PageRank at p=0.15\n(6-node toy graph)')
    axes[2].grid(True, axis='y', alpha=0.3)
    for bar, sc in zip(bars, bar_scores):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.003,
                     f'{sc:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output}')


# ─────────────────────────────────────────────────────────────
# 4.  Solver 1 — Power Iteration
# ─────────────────────────────────────────────────────────────

def pagerank_power(A, dangling_mask, p=0.15, tol=1e-10, max_iter=500):
    """
    Standard power iteration with dangling-node correction.

    Update rule:
        r_{t+1} = (1-p) * (A*r_t + dangling_mass * e) + p * e
    where e = (1/n)*1 and dangling_mass = sum of r_t over dangling nodes.

    Terminates when L1 residual < tol.
    Returns (r, history) where history is the list of L1 residuals.
    """
    n       = A.shape[0]
    r       = np.ones(n) / n
    e       = np.ones(n) / n
    history = []

    for it in range(1, max_iter + 1):
        dm    = r[dangling_mask].sum()
        r_new = (1 - p) * (A.dot(r) + dm * e) + p * e
        err   = np.abs(r_new - r).sum()
        history.append(err)
        r = r_new
        if err < tol:
            print(f'  Power iteration converged: {it} iters, L1={err:.2e}')
            break
    return r, history


# ─────────────────────────────────────────────────────────────
# 5.  Solver 2 — Closed-Form (Sparse Direct Solve)
# ─────────────────────────────────────────────────────────────

def pagerank_closed_form(A, p=0.15):
    """
    Solves the linear system [I - (1-p)*S] r = (p/n)*1 via sparse direct solve.

    Closed form: r = (p/n) * [I - (1-p)*S]^{-1} * 1

    Uses SciPy's spsolve (typically SuperLU, unless UMFPACK is available).
    Exact up to floating-point precision. Used on sub-graphs only
    (LU fill-in is prohibitive for n > ~10,000).
    """
    n = A.shape[0]
    B = sp.eye(n, format='csc', dtype=np.float64) - (1 - p) * A
    b = np.full(n, p / n, dtype=np.float64)
    r = spla.spsolve(B, b)
    r = np.abs(r)
    r /= r.sum()
    return r


# ─────────────────────────────────────────────────────────────
# 6.  Solver 3 — Monte Carlo Random-Walk Simulator
# ─────────────────────────────────────────────────────────────

def pagerank_montecarlo(out_adj, nodes, p=0.15,
                        n_walks=None, walk_length=300, seed=42):
    """
    Estimate PageRank via independent random-surfer simulations.

    Algorithm:
      For each walk:
        - Start at a uniformly random node.
        - At each step: with prob p teleport to a random node;
          otherwise follow a random out-link (teleport if dangling).
        - Accumulate visit counts.
      Normalise counts → PageRank estimate.

    This is completely independent of the matrix-based solvers:
    it uses only the raw adjacency list and no linear algebra.
    Convergence rate: O(1 / sqrt(n_walks * walk_length)).
    """
    rng = random.Random(seed)
    n   = len(nodes)
    if n_walks is None:
        n_walks = 5 * n

    nti         = {nd: i for i, nd in enumerate(nodes)}
    visit_count = np.zeros(n, dtype=np.int64)

    for _ in range(n_walks):
        cur = rng.randint(0, n - 1)
        for _ in range(walk_length):
            visit_count[cur] += 1
            outs = [nti[v] for v in out_adj.get(nodes[cur], []) if v in nti]
            if not outs or rng.random() < p:
                cur = rng.randint(0, n - 1)
            else:
                cur = rng.choice(outs)

    r = visit_count.astype(np.float64)
    r /= r.sum()
    return r


# ─────────────────────────────────────────────────────────────
# 7.  Three-Way Numerical Comparison
# ─────────────────────────────────────────────────────────────

def compare_pair(r_a, r_b, label_a='A', label_b='B'):
    """Report L1, L-inf, cosine similarity, and Spearman rank correlation."""
    l1        = np.abs(r_a - r_b).sum()
    linf      = np.abs(r_a - r_b).max()
    cos       = np.dot(r_a, r_b) / (np.linalg.norm(r_a) * np.linalg.norm(r_b))
    rank_corr = np.corrcoef(np.argsort(np.argsort(-r_a)),
                            np.argsort(np.argsort(-r_b)))[0, 1]
    print(f'  {label_a} vs {label_b}: '
          f'L1={l1:.3e}  L∞={linf:.3e}  '
          f'cos={cos:.10f}  rank-corr={rank_corr:.6f}')
    return dict(l1=l1, linf=linf, cos=cos, rank_corr=rank_corr)


def plot_method_comparison(r_pi, r_cf, r_mc, nodes, history_pi,
                           output='method_comparison.png',
                           r_pi_full=None, nodes_full=None):
    """
    Four-panel comparison figure.
    r_pi/r_cf/r_mc/nodes : 200-node subgraph vectors (scatter panels 1-2)
    history_pi           : convergence history from FULL graph (panel 3)
    r_pi_full/nodes_full : full graph data for top-15 panel (panel 4)
    """
    r_top = r_pi_full if r_pi_full is not None else r_pi
    n_top = nodes_full if nodes_full is not None else nodes

    fig = plt.figure(figsize=(15, 4))
    gs  = gridspec.GridSpec(1, 4, figure=fig)

    # Panel 1: Power iter vs closed form — should lie on y=x
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(r_cf, r_pi, s=4, alpha=0.4, color='steelblue')
    lims = [min(r_cf.min(), r_pi.min()), max(r_cf.max(), r_pi.max())]
    ax1.plot(lims, lims, 'r--', linewidth=1, label='y = x')
    ax1.set_xlabel('Closed-Form PR')
    ax1.set_ylabel('Power-Iter PR')
    ax1.set_title('Power Iter vs\nClosed Form (subgraph)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Power iter vs Monte Carlo
    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(r_pi, r_mc, s=4, alpha=0.4, color='darkorange')
    lims = [min(r_mc.min(), r_pi.min()), max(r_mc.max(), r_pi.max())]
    ax2.plot(lims, lims, 'r--', linewidth=1, label='y = x')
    ax2.set_xlabel('Power-Iter PR')
    ax2.set_ylabel('Monte Carlo PR')
    ax2.set_title('Power Iter vs\nMonte Carlo (subgraph)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Convergence trace — use full graph history
    ax3 = fig.add_subplot(gs[2])
    ax3.semilogy(range(1, len(history_pi) + 1), history_pi,
                 color='steelblue', linewidth=1.8)
    ax3.axhline(1e-10, color='red', linestyle='--',
                linewidth=1, label='Tolerance 1e-10')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('L1 Residual')
    ax3.set_title(f'Power-Iter Convergence\n(full graph, {len(history_pi)} iters)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Top-15 pages — use full graph
    ax4 = fig.add_subplot(gs[3])
    top15     = np.argsort(r_top)[::-1][:15]
    nd_labels = [str(n_top[i]) for i in top15]
    ax4.barh(nd_labels[::-1], r_top[top15][::-1],
             color='steelblue', alpha=0.85)
    ax4.set_xlabel('PageRank score')
    ax4.set_title('Top-15 Pages\n(full graph, p=0.15)')
    ax4.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output}')



# ─────────────────────────────────────────────────────────────
# 8.  Effect of p on Distribution
# ─────────────────────────────────────────────────────────────

def _gini(r):
    s = np.sort(r)
    n = len(s)
    return (2 * np.dot(np.arange(1, n + 1), s) / s.sum() - (n + 1)) / n

def _entropy(r):
    m = r > 0
    return -np.sum(r[m] * np.log(r[m]))


def study_p_effect(A, dangling_mask, p_values=None):
    """Compute PageRank for several values of p; report Gini, entropy, iters."""
    if p_values is None:
        p_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.85]
    results, conv = {}, {}
    print(f"  {'p':>5} | {'Gini':>7} | {'H(r)':>8} | "
          f"{'Top-5 sum':>10} | {'Max PR':>10} | {'Iters':>6}")
    print('  ' + '-' * 60)
    for pv in p_values:
        r, hist = pagerank_power(A, dangling_mask, p=pv, tol=1e-8)
        results[pv] = r
        conv[pv]    = len(hist)
        print(f"  {pv:>5.2f} | {_gini(r):>7.4f} | {_entropy(r):>8.4f} | "
              f"{np.partition(r, -5)[-5:].sum():>10.6f} | "
              f"{r.max():>10.6f} | {len(hist):>6}")
    return results, conv


def plot_p_effect(results, conv, output='p_effect.png'):
    pvs = sorted(results.keys())
    gi  = [_gini(results[p])    for p in pvs]
    en  = [_entropy(results[p]) for p in pvs]
    it  = [conv[p]              for p in pvs]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(pvs, gi, 'o-', color='steelblue', linewidth=2)
    axes[0].set_xlabel('p')
    axes[0].set_ylabel('Gini coefficient')
    axes[0].set_title('Rank Inequality vs p')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pvs, en, 's-', color='darkorange', linewidth=2)
    axes[1].set_xlabel('p')
    axes[1].set_ylabel('Shannon entropy H(r)')
    axes[1].set_title('Distribution Entropy vs p')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(pvs, it, '^-', color='seagreen', linewidth=2)
    axes[2].set_xlabel('p')
    axes[2].set_ylabel('Iterations to converge')
    axes[2].set_title('Convergence Speed vs p')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output}')


# ═══════════════════════════════════════════════════════════════
# PART B — EXTENSION (i): GPTBot-Like AI Crawler Prioritisation
# ═══════════════════════════════════════════════════════════════
#
# Assignment spec:
#   "Write a program that takes a small directed web graph
#    (represented as a dictionary of URLs and their outlinks)
#    and precomputed PageRank scores, then returns the top k
#    URLs to crawl based on authority. Propose one heuristic
#    to find high-quality pages that permit crawling."
#
# Inputs  : WEB_GRAPH  — dict {URL: [outlink URLs]}
#           pagerank_scores — dict {URL: float}
# Output  : top-k list of URL dicts with score breakdown
# Heuristic: Permissive-Authority Score (one heuristic)
# ─────────────────────────────────────────────────────────────

# Demo web graph: small directed graph as URL dictionary
# (as specified in the assignment)
WEB_GRAPH = {
    "https://arxiv.org/abs/2303.08774": [
        "https://arxiv.org/abs/2301.07041",
        "https://arxiv.org/abs/2210.11610",
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning)",
    ],
    "https://arxiv.org/abs/2301.07041": [
        "https://arxiv.org/abs/2303.08774",
        "https://openai.com/research/gpt-4",
    ],
    "https://arxiv.org/abs/2210.11610": [
        "https://arxiv.org/abs/2303.08774",
        "https://en.wikipedia.org/wiki/RLHF",
    ],
    "https://en.wikipedia.org/wiki/Transformer_(deep_learning)": [
        "https://arxiv.org/abs/2303.08774",
        "https://en.wikipedia.org/wiki/RLHF",
        "https://docs.python.org/3/library/math.html",
    ],
    "https://en.wikipedia.org/wiki/RLHF": [
        "https://arxiv.org/abs/2303.08774",
        "https://openai.com/research/gpt-4",
    ],
    "https://openai.com/research/gpt-4": [
        "https://arxiv.org/abs/2303.08774",
        "https://openai.com/blog/chatgpt",
    ],
    "https://openai.com/blog/chatgpt": [
        "https://openai.com/research/gpt-4",
        "https://openai.com/login",
    ],
    "https://docs.python.org/3/library/math.html": [
        "https://docs.python.org/3/",
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning)",
    ],
    "https://docs.python.org/3/": [
        "https://docs.python.org/3/library/math.html",
    ],
    "https://openai.com/login": [
        "https://openai.com/blog/chatgpt",
    ],
    "https://twitter.com/sama": [
        "https://openai.com/research/gpt-4",
    ],
    "https://reddit.com/r/MachineLearning": [
        "https://arxiv.org/abs/2303.08774",
        "https://twitter.com/sama",
    ],
}

# Simulated robots.txt rules per domain
# In production: fetch https://<domain>/robots.txt and parse with
# urllib.robotparser.RobotFileParser
ROBOTS_RULES = {
    "arxiv.org":        {"allow": True,  "note": "Permits all well-behaved crawlers"},
    "en.wikipedia.org": {"allow": True,  "note": "Open content, widely crawlable"},
    "docs.python.org":  {"allow": True,  "note": "Documentation, explicitly open"},
    "openai.com":       {"allow": True,  "note": "Permits crawlers except /login, /api"},
    "twitter.com":      {"allow": False, "note": "Restricts third-party AI crawlers"},
    "reddit.com":       {"allow": False, "note": "Restricts third-party AI crawlers"},
}

# Content-rich URL path patterns → richness score 1.0
CONTENT_RICH_PATTERNS = [
    r'/abs/',       # arXiv abstract
    r'/wiki/',      # Wikipedia article
    r'/research/',  # Research page
    r'/docs/',      # Documentation
    r'/paper/',     # Paper
    r'/blog/',      # Blog post
    r'/guide/',     # Guide
    r'/article/',   # Article
    r'/report/',    # Report
]

# Low-value URL patterns → richness score 0.0
LOW_VALUE_PATTERNS = [
    r'/login', r'/logout', r'/signup',
    r'/cart',  r'/checkout', r'/search\?',
    r'/admin', r'\?ref=',    r'/session', r'/auth',
]


def compute_pagerank_for_url_graph(web_graph, p=0.15, tol=1e-10, max_iter=500):
    """
    Compute PageRank on a URL-keyed web graph dict.
    Reuses the same sparse power-iteration solver as Part A.

    Parameters
    ----------
    web_graph : dict  {url: [outlink_urls, ...]}
    p         : teleportation probability

    Returns
    -------
    dict  {url: pagerank_score}
    """
    all_urls = sorted(set(web_graph.keys()) |
                      {v for outs in web_graph.values() for v in outs})
    n    = len(all_urls)
    idx  = {u: i for i, u in enumerate(all_urls)}

    # Build column-stochastic matrix
    rows, cols, data = [], [], []
    dangling = []
    for u in all_urls:
        i    = idx[u]
        outs = [idx[v] for v in web_graph.get(u, []) if v in idx]
        if not outs:
            dangling.append(i)
            continue
        w = 1.0 / len(outs)
        for j in outs:
            rows.append(j); cols.append(i); data.append(w)

    A  = sp.csc_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    dm = np.zeros(n, dtype=bool)
    dm[dangling] = True

    r, _ = pagerank_power(A, dm, p=p, tol=tol, max_iter=max_iter)
    return {u: float(r[idx[u]]) for u in all_urls}


def _get_domain(url):
    """Extract domain from a URL string."""
    return re.sub(r'^https?://', '', url).split('/')[0]


def _is_robots_allowed(url):
    """
    Check simulated robots.txt permission for a URL.
    In production: use urllib.robotparser to fetch the live file.
    """
    domain = _get_domain(url)
    rule   = ROBOTS_RULES.get(domain, {"allow": True})
    # Domain-specific path rule: openai.com blocks /login
    if domain == "openai.com" and "/login" in url:
        return False
    return rule["allow"]


def _content_richness(url):
    """
    Score a URL for content richness based on path patterns.
    Returns 1.0 (rich), 0.5 (neutral), or 0.0 (low-value).
    """
    for pat in LOW_VALUE_PATTERNS:
        if re.search(pat, url):
            return 0.0
    for pat in CONTENT_RICH_PATTERNS:
        if re.search(pat, url):
            return 1.0
    return 0.5


def get_top_k_urls(web_graph, pagerank_scores, k=5,
                   alpha=0.80, beta=0.20):
    """
    Return the top k URLs to crawl using the Permissive-Authority Heuristic.

    THE ONE HEURISTIC — Permissive-Authority Score
    ───────────────────────────────────────────────
    For each URL u discovered in the web graph:

        score(u) = alpha * PR(u) + beta * richness(u)

    subject to: robots.txt permits crawling u (hard constraint).

    Blocked pages receive score = 0 and are excluded from the top-k.

    Rationale
    ---------
    - PR(u) (weight 0.80): PageRank is the primary quality signal.
      High-PR pages receive links from many authoritative sources,
      reflecting collective human editorial judgment that the page
      is accurate, well-curated, and information-dense — properties
      that directly improve LLM training data quality.
    - richness(u) (weight 0.20): URL-path heuristic that rewards
      known content-rich patterns (/wiki/, /abs/, /docs/, /blog/)
      and penalises low-value transactional pages (/login, /cart).
      Acts as a tiebreaker and a lightweight spam filter without
      requiring an expensive content fetch.
    - robots.txt (hard constraint): Pages blocked by the domain's
      robots.txt are excluded entirely, ensuring compliance with
      the Web Robot Exclusion Standard and avoiding terms-of-service
      violations for the AI research organisation.

    Parameters
    ----------
    web_graph       : dict {url: [outlink_urls]}
    pagerank_scores : dict {url: float}
    k               : number of top URLs to return
    alpha           : weight on PageRank authority (default 0.80)
    beta            : weight on content-richness heuristic (default 0.20)

    Returns
    -------
    top_k    : list of dicts (ranked, robots-allowed only)
    all_results : full list including blocked pages (for reporting)
    """
    all_urls = set(web_graph.keys())
    for outs in web_graph.values():
        all_urls.update(outs)

    results = []
    for url in all_urls:
        pr       = pagerank_scores.get(url, 0.0)
        allowed  = _is_robots_allowed(url)
        richness = _content_richness(url)
        score    = (alpha * pr + beta * richness) if allowed else 0.0
        results.append({
            "url":      url,
            "pagerank": pr,
            "allowed":  allowed,
            "richness": richness,
            "score":    score,
            "blocked":  not allowed,
        })

    results.sort(key=lambda x: (-x["score"], x["blocked"], x["url"]))
    top_k = [r for r in results if not r["blocked"]][:k]
    return top_k, results


def plot_crawler_results(all_results, top_k, output='crawler_ext.png'):
    """Two-panel figure: crawl priority scores and PR vs richness scatter."""
    sorted_res = sorted(all_results, key=lambda x: -x["score"])
    labels     = [r["url"].replace("https://", "")[:42] for r in sorted_res]
    scores     = [r["score"] for r in sorted_res]
    top_urls   = {r["url"] for r in top_k}

    bar_colors = []
    for r in sorted_res:
        if r["blocked"]:
            bar_colors.append("#d9534f")    # red: robots blocked
        elif r["url"] in top_urls:
            bar_colors.append("#2c7bb6")    # blue: selected top-k
        else:
            bar_colors.append("#abd9e9")    # light: allowed, not selected

    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor='#2c7bb6', label=f'Top-{len(top_k)} selected'),
        Patch(facecolor='#abd9e9', label='Allowed, not selected'),
        Patch(facecolor='#d9534f', label='Blocked by robots.txt'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: horizontal bar — heuristic score per URL
    y = np.arange(len(labels))
    for i, (sc, cl) in enumerate(zip(scores, bar_colors)):
        axes[0].barh(len(labels) - 1 - i, sc, color=cl,
                     edgecolor='white', height=0.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels[::-1], fontsize=7.5)
    axes[0].set_xlabel("Heuristic Score = 0.80×PR + 0.20×Richness")
    axes[0].set_title("GPTBot Crawl Priority Score\n"
                      "(blue=selected, light=allowed, red=robots-blocked)")
    axes[0].legend(handles=legend_els, loc='lower right', fontsize=8)
    axes[0].grid(True, axis='x', alpha=0.3)

    # Panel 2: scatter PageRank vs richness
    for r in sorted_res:
        color  = "#d9534f" if r["blocked"] else (
                 "#2c7bb6" if r["url"] in top_urls else "#abd9e9")
        marker = 'x' if r["blocked"] else (
                 '*' if r["url"] in top_urls else 'o')
        size   = 200 if r["url"] in top_urls else 80
        axes[1].scatter(r["pagerank"], r["richness"],
                        c=color, s=size, marker=marker,
                        edgecolors='black' if not r["blocked"] else 'none',
                        linewidths=0.5,
                        zorder=3 if r["url"] in top_urls else 2)
        if r["url"] in top_urls:
            axes[1].annotate(r["url"].replace("https://", "")[:30],
                             (r["pagerank"], r["richness"]),
                             fontsize=6.5, xytext=(4, 2),
                             textcoords='offset points')

    axes[1].set_xlabel("PageRank score")
    axes[1].set_ylabel("Content-richness score (heuristic)")
    axes[1].set_title("PageRank vs Content-Richness\n"
                      "(★ = top-k selected, × = robots-blocked)")
    axes[1].legend(handles=legend_els, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {output}')


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    DATASET = 'web-Google.txt'
    P       = 0.15
    np.random.seed(42)
    random.seed(42)

    print('=' * 65)
    print('SC4052 Assignment 2, Q6 — PageRank Analysis')
    print('=' * 65)

    # ── PART A ────────────────────────────────────────────────

    # [1] Toy graph p-sweep
    print('\n[1] Toy graph: p-sweep illustration...')
    toy_adj, toy_nodes = build_toy_graph()
    p_vals, scores = toy_psweep(toy_adj, toy_nodes)
    plot_toy_psweep(p_vals, scores, toy_nodes)
    idx_015 = np.argmin(np.abs(p_vals - 0.15))
    print('  PageRank at p=0.15:')
    for nd, sc in zip(toy_nodes, scores[:, idx_015]):
        print(f'    Node {nd}: {sc:.6f}')

    # [2] Load full graph
    print(f'\n[2] Loading {DATASET}...')
    t0 = time.time()
    out_adj, nodes, n_edges = load_graph(DATASET)
    N = len(nodes)
    print(f'  Nodes: {N:,}  |  Edges: {n_edges:,}  |  {time.time()-t0:.2f}s')

    # [3] Build stochastic matrix
    print('\n[3] Building column-stochastic matrix...')
    A, nti, dangling_mask = build_stochastic(out_adj, nodes)
    print(f'  Shape: {A.shape}  |  nnz: {A.nnz:,}  |  '
          f'dangling: {dangling_mask.sum():,}')

    # [4] Power iteration (full graph)
    print(f'\n[4] Power iteration (p={P}, full {N:,}-node graph)...')
    t0 = time.time()
    r_pi, hist_pi = pagerank_power(A, dangling_mask, p=P, tol=1e-10)
    t_pi = time.time() - t0
    print(f'  Time: {t_pi:.3f}s  |  Sum: {r_pi.sum():.8f}')

    # [5] Closed-form on 200-node subgraph
    print('\n[5] Closed-form solver (200-node subgraph)...')
    SUB      = 200
    sub_nodes = nodes[:SUB]
    sub_set   = set(sub_nodes)
    sub_out   = {nd: [v for v in out_adj.get(nd, []) if v in sub_set]
                 for nd in sub_nodes}
    A_sub, nti_sub, dm_sub = build_stochastic(sub_out, sub_nodes)
    r_pi_sub, hist_sub = pagerank_power(A_sub, dm_sub, p=P, tol=1e-12)
    t0 = time.time()
    r_cf_sub = pagerank_closed_form(A_sub, p=P)
    print(f'  Closed-form time: {time.time()-t0:.3f}s')

    # [6] Monte Carlo on subgraph
    print('\n[6] Monte Carlo random-walk solver (200-node subgraph)...')
    t0 = time.time()
    r_mc_sub = pagerank_montecarlo(sub_out, sub_nodes, p=P,
                                   n_walks=50 * SUB, walk_length=300)
    print(f'  MC time: {time.time()-t0:.3f}s')

    # [7] Numerical comparison
    print('\n[7] Numerical comparison (200-node subgraph, p=0.15):')
    compare_pair(r_pi_sub, r_cf_sub, 'PowerIter',  'ClosedForm')
    compare_pair(r_pi_sub, r_mc_sub, 'PowerIter',  'MonteCarlo')
    compare_pair(r_cf_sub, r_mc_sub, 'ClosedForm', 'MonteCarlo')
    plot_method_comparison(r_pi_sub, r_cf_sub, r_mc_sub,
                           sub_nodes, hist_pi,
                           'method_comparison.png',
                           r_pi_full=r_pi, nodes_full=nodes)

    # [8] Top-20 pages on full graph
    print(f'\n[8] Top-20 pages (full {N:,}-node graph, p={P}):')
    top20  = np.argsort(r_pi)[::-1][:20]
    in_deg = defaultdict(int)
    for nd, outs in out_adj.items():
        for v in outs:
            if v in nti:
                in_deg[v] += 1
    print(f'  {"Rk":<4} {"Node":>10} {"PageRank":>12} {"In-deg":>8} {"Out-deg":>8}')
    print('  ' + '-' * 48)
    for rk, idx in enumerate(top20, 1):
        nd = nodes[idx]
        print(f'  {rk:<4} {nd:>10} {r_pi[idx]:>12.8f} '
              f'{in_deg[nd]:>8} {len(out_adj.get(nd, [])):>8}')

    # [9] Effect of p — run on 50k subgraph for speed
    # (full 875k-node graph × 7 p-values × ~25 iters each is slow)
    print('\n[9] Effect of p on distribution (50k-node subgraph)...')
    P_SUB       = 50000
    psub_nodes  = nodes[:P_SUB]
    psub_set    = set(psub_nodes)
    psub_out    = {nd: [v for v in out_adj.get(nd, []) if v in psub_set]
                   for nd in psub_nodes}
    A_psub, _, dm_psub = build_stochastic(psub_out, psub_nodes)
    p_results, p_conv  = study_p_effect(A_psub, dm_psub)
    plot_p_effect(p_results, p_conv)

    # [10] Log-log rank distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_r = np.sort(r_pi)[::-1]
    ax.loglog(np.arange(1, N + 1), sorted_r,
              color='steelblue', linewidth=1.5)
    ax.set_xlabel('Rank (log)')
    ax.set_ylabel('PageRank score (log)')
    ax.set_title(f'PageRank Score Distribution\nweb-Google.txt (n={N:,}, p={P})')
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('rank_dist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: rank_dist.png')

    # ── PART B — EXTENSION (i) ────────────────────────────────

    print('\n' + '=' * 65)
    print('PART B — Extension (i): GPTBot-Like Crawler')
    print('=' * 65)

    # [11-12] Compute PageRank on URL graph
    print('\n[11] Computing PageRank on demo URL web graph...')
    pagerank_scores = compute_pagerank_for_url_graph(WEB_GRAPH, p=P)
    print('\n  PageRank scores:')
    for url, pr in sorted(pagerank_scores.items(), key=lambda x: -x[1]):
        print(f'    {pr:.6f}  {url.replace("https://", "")}')

    # [13-14] Apply Permissive-Authority Heuristic
    print('\n[12] Applying Permissive-Authority Heuristic (top k=5)...')
    top_k, all_results = get_top_k_urls(WEB_GRAPH, pagerank_scores, k=5)

    print('\n  Top-5 URLs to crawl:')
    print(f'  {"Rk":<5} {"PageRank":<10} {"Richness":<10} '
          f'{"Score":<10} {"Robots":<8} URL')
    print('  ' + '-' * 85)
    for rank, r in enumerate(top_k, 1):
        status = 'ALLOW' if not r['blocked'] else 'BLOCK'
        print(f'  {rank:<5} {r["pagerank"]:<10.6f} {r["richness"]:<10.1f} '
              f'{r["score"]:<10.6f} {status:<8} '
              f'{r["url"].replace("https://", "")}')

    print('\n  Blocked pages (excluded from crawl queue):')
    for r in all_results:
        if r['blocked']:
            domain = _get_domain(r['url'])
            note   = ROBOTS_RULES.get(domain, {}).get('note', '')
            print(f'    [BLOCKED]  {r["url"].replace("https://", "")}'
                  f'  (PR={r["pagerank"]:.6f}, {note})')

    # [15] Figure
    print('\n[13] Generating crawler figure...')
    plot_crawler_results(all_results, top_k, 'crawler_ext.png')

    print('\n  Why high-PageRank pages yield better LLM training data:')
    print("""
    1. AUTHORITY AS QUALITY PROXY: High-PR pages receive links from
       many authoritative sources, encoding collective editorial
       judgement that the page is accurate and well-curated.

    2. FACTUAL DENSITY: Academic papers, encyclopaedia articles, and
       documentation contain dense structured prose — high training
       signal per token vs. boilerplate HTML on utility pages.

    3. NATURAL SPAM FILTER: Low-quality auto-generated or duplicate
       content receives few authoritative in-links → low PageRank.
       Prioritising by PR filters junk without a content classifier.

    4. CONCEPT IMPORTANCE: Entities and facts on high-PR pages are
       those humans collectively consider most important — exactly
       what an LLM should learn well.
    """)

    print('=' * 65)
    print('All done. Figures saved:')
    print('  toy_psweep.png  method_comparison.png  rank_dist.png')
    print('  p_effect.png    crawler_ext.png')
    print('=' * 65)
