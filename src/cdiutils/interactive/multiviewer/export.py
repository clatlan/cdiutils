"""
Ordered, bounded, cancellable frame export for :class:`MultiVolumeViewer`.

Frames are *produced* sequentially in the main thread, *rendered* to PNG
either sequentially in a worker thread (``safe``) or in a process pool
(``fast`` / ``process``), and *written* strictly in order.

When a ``base`` scene is given, frames are small *deltas* (a new camera, or a
few replaced traces). The base is sent to each worker once per export
(through a temporary file) and the workers stay alive between exports, so a
long export does not pickle the whole scene for every frame. Workers also
keep the decoded triangles of unchanged traces for the software renderer.

GIF files are written with Pillow and MP4 files with the ffmpeg binary that
ships with ``imageio-ffmpeg``: no system ffmpeg is needed.
"""

import asyncio
import atexit
import concurrent.futures
import io
import multiprocessing
import os
import pickle
import tempfile
import threading
from pathlib import Path

import numpy as np

FORMATS = ("mp4", "gif", "png")
REPEAT = "__repeat_previous_frame__"  # produce(k) result: reuse frame k-1


def output_path(raw: str, fmt: str, default_stem: str) -> Path:
    """Output path from a name or a path, with the right extension.

    Accepts ``name``, ``name.mp4``, ``folder/sub/name``, ``~/name``, an
    absolute path, or a folder ending with ``/`` (then ``default_stem`` is
    used). Missing folders are created when the file is written. Dots in
    the name are kept (``run_1.5bar`` -> ``run_1.5bar.mp4``). For
    ``fmt="png"`` the result is the folder that receives the frames.
    """
    raw = (raw or default_stem).strip()
    p = Path(raw).expanduser()
    if raw.endswith(("/", "\\")) or (
        p.exists() and p.is_dir() and fmt != "png"
    ):
        p = p / default_stem
    if p.suffix.lower() in (".mp4", ".gif", ".png", ".html"):
        p = p.with_suffix("")
    return p if fmt == "png" else p.with_name(f"{p.name}.{fmt}")


def _png_to_rgb_on_white(png_bytes):
    import imageio.v2 as iio

    img = np.asarray(iio.imread(io.BytesIO(png_bytes)))
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (
            (img[..., :3].astype(np.float32) * a + 255.0 * (1.0 - a))
            .round()
            .astype(np.uint8)
        )
    return img


class FrameSink:
    """Writes PNG frames to MP4 (imageio-ffmpeg), GIF (Pillow) or a PNG folder.

    Transparent pixels are composited on white for MP4/GIF.
    """

    def __init__(self, out_path, fmt: str, fps: int):
        self.fmt = str(fmt).lower()
        if self.fmt not in FORMATS:
            raise ValueError(
                f"Unsupported format {fmt!r}; use one of {FORMATS}"
            )
        self.out_path = Path(out_path)
        self.fps = max(1, int(fps))
        self.n = 0
        self._writer = None
        self._tmp = None
        if self.fmt == "mp4":
            try:
                import imageio.v2 as iio
                import imageio_ffmpeg  # noqa: F401  (availability check)
            except Exception as e:
                raise RuntimeError(
                    "MP4 export requires imageio-ffmpeg: pip install imageio-ffmpeg"
                ) from e
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = iio.get_writer(
                str(self.out_path),
                fps=self.fps,
                codec="libx264",
                macro_block_size=2,
            )
        elif self.fmt == "gif":
            self._tmp = tempfile.TemporaryDirectory()
            self._dir = Path(self._tmp.name)
        else:
            self._dir = self.out_path
            self._dir.mkdir(parents=True, exist_ok=True)

    def add(self, png_bytes: bytes):
        if self.fmt == "mp4":
            self._writer.append_data(_png_to_rgb_on_white(png_bytes))
        elif self.fmt == "gif":
            import imageio.v2 as iio

            iio.imwrite(
                self._dir / f"frame_{self.n:06d}.png",
                _png_to_rgb_on_white(png_bytes),
            )
        else:
            (self._dir / f"frame_{self.n:06d}.png").write_bytes(png_bytes)
        self.n += 1

    def close(self, encode: bool = True):
        try:
            if self.fmt == "mp4" and self._writer is not None:
                self._writer.close()
            elif self.fmt == "gif" and encode and self.n:
                self._encode_gif()
        finally:
            if self._tmp is not None:
                self._tmp.cleanup()
                self._tmp = None

    def _encode_gif(self):
        """Assemble the frames with Pillow (one shared adaptive palette)."""
        from PIL import Image

        files = sorted(self._dir.glob("frame_*.png"))
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(files[0]) as im0:
            first = im0.convert("RGB")
        # Palette from a few frames spread over the animation, so colours
        # that appear late are kept and the palette does not flicker.
        picks = files[:: max(1, len(files) // 6)][:6]
        thumbs = []
        for f in picks:
            with Image.open(f) as im:
                thumbs.append(im.convert("RGB"))
        sheet = Image.new("RGB", (first.width, first.height * len(thumbs)))
        for i, t in enumerate(thumbs):
            sheet.paste(t, (0, i * first.height))
        pal = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)

        def frames():
            for f in files[1:]:
                with Image.open(f) as im:
                    yield im.convert("RGB").quantize(
                        palette=pal, dither=Image.Dither.FLOYDSTEINBERG
                    )

        first.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG).save(
            self.out_path,
            save_all=True,
            append_images=frames(),
            duration=max(10, int(round(1000 / self.fps))),
            loop=0,
            optimize=False,
            disposal=1,
        )


# --------------------------------------------------------------------------
# Scene deltas
# --------------------------------------------------------------------------


def merge_delta(base, delta):
    """Full figure JSON from a base scene and a frame delta.

    ``delta`` may hold ``camera`` (replaces ``layout.scene.camera``),
    ``traces`` (``{index: trace_json}``, replaces those traces) and
    ``annotations`` (replaces ``layout.annotations``, e.g. a frame label).
    """
    data = list(base.get("data") or [])
    layout = base.get("layout") or {}
    for i, tr in (delta.get("traces") or {}).items():
        data[int(i)] = tr
    if delta.get("annotations") is not None:
        layout = dict(layout, annotations=delta["annotations"])
    if delta.get("camera") is not None:
        layout = dict(layout)
        layout["scene"] = dict(layout.get("scene") or {})
        layout["scene"]["camera"] = delta["camera"]
    return {"data": data, "layout": layout}


class _Scene:
    """A base scene and the decoded primitives of its unchanged traces."""

    def __init__(self, base):
        self.base = base
        self.prims = {}

    def figure(self, delta, software):
        fig = merge_delta(self.base, delta)
        if software:
            from .raster import extract_trace

            changed = {int(i) for i in (delta.get("traces") or {})}
            prims = []
            for i, tr in enumerate(fig["data"]):
                if i in changed:
                    prims.append(extract_trace(tr))
                    continue
                if i not in self.prims:
                    self.prims[i] = extract_trace(tr)
                prims.append(self.prims[i])
            fig["_mv_prims"] = prims
        return fig


_WORKER_SCENES: dict = {}  # (scene file, export id) -> _Scene (per process)


def _is_software(render_args):
    return len(render_args) > 3 and str(
        render_args[3] or "software"
    ).lower() in (
        "software",
        "auto",
    )


def render_delta(scene, delta, render_fn, render_args):
    """Worker entry point: render one delta frame.

    ``scene`` is ``(path, export_id)``; the pickled base scene at ``path`` is
    loaded once per process and export.
    """
    scene_path, _ = scene
    sc = _WORKER_SCENES.get(scene)
    if sc is None:
        with open(scene_path, "rb") as f:
            sc = _Scene(pickle.load(f))
        while len(_WORKER_SCENES) >= 2:  # keep memory bounded
            _WORKER_SCENES.pop(next(iter(_WORKER_SCENES)))
        _WORKER_SCENES[scene] = sc
    return render_fn(sc.figure(delta, _is_software(render_args)), *render_args)


# --------------------------------------------------------------------------
# Persistent process pool
# --------------------------------------------------------------------------

_POOL_LOCK = threading.Lock()
_POOL: dict = {"executor": None, "key": None}


def get_pool(workers: int):
    """Shared spawn process pool, created on first use and reused.

    Recreated when the number of workers or the start method changes, or
    after it broke. Closed by :func:`shutdown_pool` (also at exit).
    """
    ctx_name = os.environ.get("MV_MP_CONTEXT", "spawn")
    key = (ctx_name, max(1, int(workers)))
    with _POOL_LOCK:
        ex = _POOL["executor"]
        if ex is not None and (
            _POOL["key"] != key or getattr(ex, "_broken", False)
        ):
            ex.shutdown(wait=False, cancel_futures=True)
            ex = None
        if ex is None:
            ex = concurrent.futures.ProcessPoolExecutor(
                max_workers=key[1],
                mp_context=multiprocessing.get_context(ctx_name),
            )
            _POOL["executor"], _POOL["key"] = ex, key
        return ex


def shutdown_pool(wait: bool = True):
    """Stop the shared render workers (they restart on the next export)."""
    with _POOL_LOCK:
        ex, _POOL["executor"], _POOL["key"] = _POOL["executor"], None, None
    if ex is not None:
        ex.shutdown(wait=wait, cancel_futures=True)


def pool_alive() -> bool:
    return _POOL["executor"] is not None


atexit.register(shutdown_pool, False)


def make_executor(mode: str, workers: int):
    """``safe`` -> None (sequential in a thread); ``fast``/``process`` -> the
    shared persistent process pool."""
    if mode == "safe":
        return None
    return get_pool(workers)


async def run_ordered_export(
    n_frames,
    produce,
    render_fn,
    render_args,
    sink,
    *,
    mode="safe",
    workers=1,
    in_flight=1,
    is_cancelled=lambda: False,
    on_progress=None,
    base=None,
):
    """Produce, render and write ``n_frames`` frames in order.

    ``produce(k)`` returns a figure JSON dict (or, with ``base``, a delta
    for :func:`merge_delta`), ``None`` to skip frame k, or :data:`REPEAT`
    to write the previous frame again (no rendering).
    ``render_fn(fig_json, *render_args)`` returns PNG bytes; it must be a
    module-level function when a process pool is used.
    Returns the number of frames written.
    """
    loop = asyncio.get_running_loop()
    executor = make_executor(mode, workers)
    limit = 1 if executor is None else max(1, int(in_flight))
    tasks: dict = {}
    pending: dict = {}
    nxt = 0
    scene_path = None
    local = None
    export_id = os.urandom(8).hex()
    if base is not None:
        if executor is None:
            local = _Scene(base)
        else:
            fd, scene_path = tempfile.mkstemp(
                prefix="mv_scene_", suffix=".pkl"
            )
            with os.fdopen(fd, "wb") as f:
                pickle.dump(base, f, protocol=pickle.HIGHEST_PROTOCOL)
    software = _is_software(render_args)

    def _render_local(frame):
        if local is not None:
            frame = local.figure(frame, software)
        return render_fn(frame, *render_args)

    async def _render(k, frame):
        if executor is None:
            return k, await asyncio.to_thread(_render_local, frame)
        if scene_path is not None:
            return k, await loop.run_in_executor(
                executor,
                render_delta,
                (scene_path, export_id),
                frame,
                render_fn,
                render_args,
            )
        return k, await loop.run_in_executor(
            executor, render_fn, frame, *render_args
        )

    last = None

    def _flush():
        nonlocal nxt, last
        while nxt in pending:
            png = pending.pop(nxt)
            if png is REPEAT:
                png = last
            if png:
                sink.add(png)
                last = png
            nxt += 1
            if on_progress is not None:
                on_progress(nxt, n_frames)

    async def _collect():
        done, _ = await asyncio.wait(
            tasks.keys(), return_when=asyncio.FIRST_COMPLETED
        )
        for t in done:
            tasks.pop(t)
            k, png = t.result()  # re-raises render errors
            pending[k] = png
        _flush()

    try:
        for k in range(int(n_frames)):
            if is_cancelled():
                break
            fig_json = produce(k)
            if fig_json is None or fig_json is REPEAT:
                pending[k] = fig_json
                _flush()
            else:
                tasks[asyncio.ensure_future(_render(k, fig_json))] = k
            while len(tasks) >= limit:
                await _collect()
            await asyncio.sleep(0)
        while tasks and not is_cancelled():
            await _collect()
    finally:
        for t in tasks:
            t.cancel()
        if scene_path is not None:
            try:
                os.unlink(scene_path)  # workers already hold it in memory
            except OSError:
                pass
    return sink.n
