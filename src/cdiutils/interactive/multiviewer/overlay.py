"""
Matplotlib overlay for the software renderer: axes box, grid, ticks, axis
titles, colorbars, 3D annotations and 2D (paper) annotations, composed with
the 3D image from :mod:`.raster` into a PNG.

Only numpy and matplotlib are needed (Agg backend, no display).
"""

import io
import re

import numpy as np

from .raster import colorscale_lut, render_scene
from .rendering import parse_color

AXES = ("xaxis", "yaxis", "zaxis")


def _rgba(value, default=(0, 0, 0, 1)):
    if value is None:
        return default
    t = str(value).strip()
    m = re.match(r"rgba\(([^)]*)\)", t)
    if m:
        v = [float(x) for x in m.group(1).split(",")]
        return (
            v[0] / 255,
            v[1] / 255,
            v[2] / 255,
            v[3] if len(v) > 3 else 1.0,
        )
    return (*parse_color(t, default[:3]), 1.0)


def _template_layout(layout):
    tpl = layout.get("template") or {}
    return tpl.get("layout") or {} if isinstance(tpl, dict) else {}


def _strip_html(text):
    t = re.sub(r"<br\s*/?>", "\n", str(text or ""))
    t = re.sub(r"<sup>(.*?)</sup>", r"^\1", t)
    return re.sub(r"<[^>]+>", "", t)


def _title_text(obj):
    t = (obj or {}).get("title")
    if isinstance(t, dict):
        return t.get("text") or ""
    return t or ""


def _font(spec, fallback_size, scale, color):
    spec = spec or {}
    fam = str(spec.get("family") or "DejaVu Sans")
    bold = "bold" in fam.lower()
    fam = re.sub(r"(?i)\s*bold", "", fam).strip() or "DejaVu Sans"
    return dict(
        family=fam
        if fam.lower() in ("dejavu sans", "sans-serif", "arial")
        else "DejaVu Sans",
        weight="bold" if bold else "normal",
        size=float(spec.get("size") or fallback_size)
        * scale
        * 0.75,  # px -> pt at 100 dpi
        color=spec.get("color") or color,
    )


def _ticks(lo, hi, n=4):
    from matplotlib.ticker import MaxNLocator

    vals = MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    span = hi - lo
    return [v for v in vals if lo - 1e-9 * span <= v <= hi + 1e-9 * span]


def _fmt(v, step):
    if (step and abs(step) < 1e-3) or abs(v) >= 1e5:
        return f"{v:.2g}"
    dec = max(0, -int(np.floor(np.log10(abs(step))))) if step else 0
    return f"{v:.{min(dec, 6)}f}"


def _draw_axes(ax, cam, bounds, scene, scale, text_color, front):
    """Grid on the back faces (front=False) or axis lines, ticks and titles (front=True)."""
    lo, hi = bounds
    eye_plot = cam.mid + cam.eye / cam.scale
    back = np.where(
        eye_plot > cam.mid, lo, hi
    )  # coordinate of the back face per axis
    axes = [dict(scene.get(a) or {}) for a in AXES]

    def proj(P):
        x, y, _ = cam.project(np.asarray(P, float))
        return x, y

    ticks = [
        _ticks(lo[i], hi[i], int(axes[i].get("nticks") or 4)) for i in range(3)
    ]
    if not front:
        for a in range(3):
            if (
                not axes[a].get("visible", True)
                and axes[a].get("visible") is not None
            ):
                continue
            for b in range(3):
                if b == a:
                    continue
                c = 3 - a - b
                if not axes[b].get("showgrid", True):
                    continue
                gcol = _rgba(axes[b].get("gridcolor"), (0, 0, 0, 0.2))
                lw = float(axes[b].get("gridwidth") or 1) * scale * 0.7
                for v in ticks[b]:
                    P = np.zeros((2, 3))
                    P[:, a] = back[a]
                    P[:, b] = v
                    P[:, c] = [lo[c], hi[c]]
                    x, y = proj(P)
                    ax.plot(
                        x,
                        y,
                        color=gcol[:3],
                        alpha=gcol[3],
                        lw=lw,
                        zorder=1,
                        solid_capstyle="butt",
                    )
        return
    ctr = np.array(proj(0.5 * (lo + hi)[None]))[:, 0]
    for a in range(3):
        if axes[a].get("visible") is False:
            continue
        b, c = [k for k in range(3) if k != a]
        cands = []
        for vb in (lo[b], hi[b]):
            for vc in (lo[c], hi[c]):
                on_back = (vb == back[b]) + (vc == back[c])
                if (
                    on_back == 1
                ):  # silhouette edge: one back face, one front face
                    P = np.zeros((2, 3))
                    P[:, a] = [lo[a], hi[a]]
                    P[:, b], P[:, c] = vb, vc
                    x, y = proj(P)
                    cands.append((P, x, y))
        if not cands:
            continue
        dx = np.diff(cands[0][1])[0]
        dy = np.diff(cands[0][2])[0]
        vertical = abs(dy) > abs(dx)
        P, x, y = (
            min(cands, key=lambda e: np.mean(e[1]))
            if vertical
            else max(cands, key=lambda e: np.mean(e[2]))
        )
        lc = _rgba(axes[a].get("linecolor"), (*parse_color(text_color), 1))
        if axes[a].get("showline", True):
            ax.plot(
                x,
                y,
                color=lc[:3],
                lw=float(axes[a].get("linewidth") or 2) * scale * 0.75,
                zorder=5,
                solid_capstyle="round",
            )
        # outward direction on screen (perpendicular to the edge, away from the box centre)
        ex, ey = x[1] - x[0], y[1] - y[0]
        n = np.array([-ey, ex])
        n /= np.linalg.norm(n) or 1.0
        mid = np.array([x.mean(), y.mean()])
        if np.dot(mid - ctr, n) < 0:
            n = -n
        tfont = _font(axes[a].get("tickfont"), 12, scale, text_color)
        tl = float(axes[a].get("ticklen") or 6) * scale * 0.6
        step = (ticks[a][1] - ticks[a][0]) if len(ticks[a]) > 1 else 0
        for v in ticks[a]:
            Q = P[0].copy()
            Q[a] = v
            tx, ty = proj(Q[None])
            tx, ty = float(tx[0]), float(ty[0])
            ax.plot(
                [tx, tx + n[0] * tl],
                [ty, ty + n[1] * tl],
                color=lc[:3],
                lw=1.2 * scale,
                zorder=5,
            )
            off = tl + 0.9 * tfont["size"] * 1.4
            ax.text(
                tx + n[0] * off,
                ty + n[1] * off,
                _fmt(v, step),
                ha="center",
                va="center",
                zorder=6,
                **tfont,
            )
        title = _strip_html(_title_text(axes[a]))
        if title:
            f = _font(
                (axes[a].get("title") or {}).get("font")
                if isinstance(axes[a].get("title"), dict)
                else None,
                18,
                scale,
                text_color,
            )
            off = tl + 0.9 * tfont["size"] * 1.4 + 2.2 * f["size"] * 1.3
            ang = np.degrees(np.arctan2(-ey, ex))
            if ang > 90:
                ang -= 180
            elif ang < -90:
                ang += 180
            ax.text(
                mid[0] + n[0] * off,
                mid[1] + n[1] * off,
                title,
                ha="center",
                va="center",
                rotation=ang,
                rotation_mode="anchor",
                zorder=6,
                **f,
            )


def _colorbars(fig_json):
    out = []
    for tr in fig_json.get("data", []):
        if tr.get("visible", True) is False or not tr.get("showscale"):
            continue
        if str(tr.get("type", "")).lower() not in ("mesh3d", "surface"):
            continue
        cb = dict(tr.get("colorbar") or {})
        out.append(
            dict(
                colorscale=tr.get("colorscale"),
                cmin=tr.get("cmin"),
                cmax=tr.get("cmax"),
                cb=cb,
            )
        )
    return out


def _fit_ticks(fig, cax, width_px):
    """Shrink colorbar tick labels that would run past the image edge."""
    r = fig.canvas.get_renderer()
    for _ in range(8):
        labels = [t for t in cax.get_yticklabels() if t.get_text()]
        if not labels:
            return
        right = max(t.get_window_extent(r).x1 for t in labels)
        if right <= width_px - 2:
            return
        size = labels[0].get_fontsize() * 0.88
        cax.tick_params(labelsize=size)


def _fit_title(fig, text, x0_px, x1_px):
    """Keep a colorbar title within ``[x0_px, x1_px]`` (its column, so
    neighbouring titles do not overlap): break it at " | " and before a
    trailing "(unit)", then shrink the font if it is still too wide."""
    r = fig.canvas.get_renderer()

    def fits():
        bb = text.get_window_extent(r)
        return bb.x0 >= x0_px - 0.5 and bb.x1 <= x1_px + 0.5

    if fits():
        return
    label = text.get_text().replace(" | ", "\n")
    if " (" in label.splitlines()[-1]:
        head, _, tail = label.rpartition(" (")
        label = f"{head}\n({tail}"
    text.set_text(label)
    text.set_multialignment("center")
    for _ in range(10):
        if fits():
            return
        text.set_fontsize(text.get_fontsize() * 0.85)


def render_software_png(
    fig_json,
    width,
    height,
    scale=1,
    fontsize=14,
    smooth=False,
    flat=False,
    dpi=100,
):
    """Render a Plotly 3D figure JSON to PNG bytes with numpy + matplotlib."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.figure import Figure

    layout = dict(fig_json.get("layout") or {})
    tpl = _template_layout(layout)
    scene = dict(layout.get("scene") or {})
    s = max(1, int(scale))
    W, H = int(width) * s, int(height) * s
    paper = _rgba(
        layout.get("paper_bgcolor") or tpl.get("paper_bgcolor"), (1, 1, 1, 1)
    )
    scene_bg = _rgba(
        scene.get("bgcolor") or (tpl.get("scene") or {}).get("bgcolor"), paper
    )
    dark = sum(paper[:3]) < 1.5
    text_color = (layout.get("font") or {}).get("color") or (
        "#e6eefc" if dark else "#000000"
    )
    cbs = _colorbars(fig_json)
    cb_frac = min(0.5, 0.15 * len(cbs))
    Ws = int(round(W * (1 - cb_frac)))

    rgba, cam, prims, bounds = render_scene(
        fig_json, Ws, H, smooth=smooth, flat=flat
    )

    fig = Figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=paper[:3])
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, Ws / W, 1])
    ax.set_xlim(0, Ws)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.add_patch(
        matplotlib.patches.Rectangle(
            (0, 0), Ws, H, color=scene_bg[:3], zorder=0
        )
    )
    show_axes = any(
        (scene.get(a) or {}).get("visible", True) is not False for a in AXES
    )
    if show_axes:
        _draw_axes(ax, cam, bounds, scene, s, text_color, front=False)
    ax.imshow(rgba, extent=(0, Ws, H, 0), interpolation="nearest", zorder=3)
    if show_axes:
        _draw_axes(ax, cam, bounds, scene, s, text_color, front=True)

    for a in scene.get("annotations") or []:
        if a.get("visible") is False:
            continue
        try:
            P = np.array(
                [
                    [
                        float(a.get("x", 0)),
                        float(a.get("y", 0)),
                        float(a.get("z", 0)),
                    ]
                ]
            )
        except (TypeError, ValueError):
            continue
        x, y, d = cam.project(P)
        if not np.isfinite(x[0]) or d[0] <= 0:
            continue
        f = _font(a.get("font"), 14, s, text_color)
        if "<b>" in str(a.get("text", "")):
            f["weight"] = "bold"
        ax.text(
            x[0],
            y[0],
            _strip_html(a.get("text")),
            ha="center",
            va="center",
            zorder=7,
            **f,
        )

    for i, cb in enumerate(cbs):
        spec = cb["cb"]
        length = float(spec.get("len") or 0.7)
        x0 = (Ws + W * 0.15 * i + 0.02 * W) / W
        cax = fig.add_axes([x0, 0.5 - length / 2, 0.028, length])
        lut = colorscale_lut(cb["colorscale"])
        lo = float(cb["cmin"]) if cb["cmin"] is not None else 0.0
        hi = float(cb["cmax"]) if cb["cmax"] is not None else 1.0
        if hi <= lo:
            hi = lo + 1e-12
        bar = ColorbarBase(
            cax,
            cmap=ListedColormap(lut),
            norm=Normalize(lo, hi),
            orientation="vertical",
        )
        tf = _font(spec.get("tickfont"), 14, s, text_color)
        if spec.get("tickvals") is not None:
            bar.set_ticks(list(spec["tickvals"]))
            if spec.get("ticktext") is not None:
                bar.set_ticklabels([_strip_html(t) for t in spec["ticktext"]])
        cax.tick_params(
            labelsize=tf["size"],
            colors=tf["color"],
            width=1.2 * s,
            length=4 * s,
        )
        for lab in cax.get_yticklabels():
            lab.set_fontweight(tf["weight"])
        _fit_ticks(fig, cax, W)
        outline = _rgba(
            spec.get("outlinecolor"), (*parse_color(text_color), 1)
        )
        bar.outline.set_edgecolor(outline[:3])
        bar.outline.set_linewidth(
            float(spec.get("outlinewidth") or 1) * s * 0.75
        )
        title = _strip_html(_title_text(spec))
        if title:
            f = _font(
                (spec.get("title") or {}).get("font")
                if isinstance(spec.get("title"), dict)
                else None,
                16,
                s,
                text_color,
            )
            t = cax.set_title(title, pad=6 * s, **f)
            col = 0.15 * W  # colorbar column width
            cx = (cax.get_position().x0 + cax.get_position().x1) / 2 * W
            _fit_title(fig, t, max(0, cx - col / 2), min(W, cx + col / 2))

    for a in (
        layout.get("annotations") or []
    ):  # 2D (paper) annotations, e.g. the condition label
        if a.get("visible") is False:
            continue
        f = _font(a.get("font"), 16, s, text_color)
        fig.text(
            float(a.get("x", 0.02)),
            float(a.get("y", 0.98)),
            _strip_html(a.get("text")),
            ha={"left": "left", "right": "right"}.get(
                a.get("xanchor"), "center"
            ),
            va={"top": "top", "bottom": "bottom"}.get(
                a.get("yanchor"), "center"
            ),
            **f,
        )
    title = _strip_html(_title_text(layout))
    if title:
        fig.text(
            0.5,
            0.97,
            title,
            ha="center",
            va="top",
            **_font(
                (layout.get("title") or {}).get("font")
                if isinstance(layout.get("title"), dict)
                else None,
                20,
                s,
                text_color,
            ),
        )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=paper[:3])
    return buf.getvalue()
