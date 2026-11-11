"""The figure registry (docs/figures_plan.md §4).

Each figure is registered with what it answers, a takeaway that is formatted from the facts its
builder returns, and the step it led to. A builder returns ``(figure, facts)``; it never changes
a decision. ``kind`` is "process" (what the rebuild did) or "correction" (a difference from the
submitted project, used in docs/defense_notes.md).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

STAGES = ("0", "1", "2", "3", "3b", "4", "5", "6", "6b", "7", "8", "9", "correction")
STAGE_NAMES = {
    "0": "Raw data intake", "1": "Infrastructure, splits and leakage contract",
    "2": "Ingestion and the UTC time base", "3": "Exploratory analysis",
    "3b": "Data quality before and after cleaning", "4": "Logger clocks and hourly data",
    "5": "Cleaning", "6": "Target and features", "6b": "Weather forecasts (GFS)",
    "7": "Feature selection", "8": "Evaluation layer", "9": "Light models and the runner",
    "correction": "Corrections of the submitted project (defense notes)",
}


@dataclass(frozen=True)
class FigureSpec:
    id: str
    stage: str
    slug: str
    title: str
    question: str
    description: str
    takeaway: str
    led_to: str
    inputs: tuple[str, ...]
    build: Callable = field(compare=False, repr=False)
    kind: str = "process"

    @property
    def stage_dir(self) -> str:
        return "correction" if self.stage == "correction" else f"stage_{self.stage.zfill(2)}"

    @property
    def filename(self) -> str:
        return f"{self.id}_{self.slug}.png"


REGISTRY: list[FigureSpec] = []


def register(id, stage, slug, title, question, description, takeaway, led_to, inputs=(),
             kind="process"):
    assert stage in STAGES, stage
    assert kind in ("process", "correction"), kind
    assert (kind == "correction") == (stage == "correction"), id

    def wrap(fn):
        REGISTRY.append(FigureSpec(id, stage, slug, title, question, description, takeaway,
                                   led_to, tuple(inputs), fn, kind))
        return fn
    return wrap


# Figures made by the stage producers before the figure book. Listed in the catalog with their
# producer; they are checked by their own stage tests, not rebuilt here.
EXISTING = [
    ("3", "outputs/03_eda/figures", "python -m solarpv.eda", "fig*.png"),
    ("3b", "outputs/03b_quality/figures", "python -m solarpv.eda.quality", "q*.png"),
    ("6", "outputs/06_features", "python -m solarpv.features.decide_k", "d11_l04.png"),
    ("6b", "outputs/06b_gfs", "python -m solarpv.nwp.check", "value_check.png"),
    ("7", "outputs/07_feature_selection/figures", "python -m solarpv.selection", "fig*.png"),
    ("9", "outputs/runs/20261107T135938Z_dev/xgb/figures",
     "python -m solarpv.pipeline run --mode dev", "*.png"),
]


def load_all() -> list[FigureSpec]:
    """Import every stage module so that its figures register, in stage order."""
    from solarpv.figures import correction, s00_s02, s04_s05, s06, s07_s09  # noqa: F401
    order = {s: i for i, s in enumerate(STAGES)}
    ids = [f.id for f in REGISTRY]
    assert len(ids) == len(set(ids)), "duplicate figure id"
    return sorted(REGISTRY, key=lambda f: (order[f.stage], f.id))
