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
    *(("9", f"outputs/runs/20261111T160714Z_dev/{step}/figures",
       "python -m solarpv.pipeline run --mode dev", "*.png")
      for step in ("bilstm", "tft96", "tft168", "tftbase", "patchtst")),
]


# The figures of each work item of a stage that has work items (docs/stageN_plan.md), for the
# per-task figure sheets (docs/figures_plan.md, rule 7). A stage without work items is one task,
# and so are the correction figures made with the figure book (OP-56). When a task is completed,
# its figures are listed here and its sheet is written.
WORK_ITEMS = {
    "9": {
        "W1": ("The runner, the baselines and the evaluate step", ("S09-01", "S09-33")),
        "W2": ("The XGB step and its development run",
               ("S09-02", "S09-03", "S09-04", "S09-05", "S09-31", "S09-32")),
        "W3": ("The datasets of the sequence models", ("S09-06",)),
        "W4": ("The five neural models",
               ("S09-07", "S09-08", "S09-09", "S09-10", "S09-11", "S09-12", "S09-25",
                "S09-26", "S09-27", "S09-28", "S09-29", "S09-30", "C-07", "C-08", "C-09")),
        "W5": ("Select, explain, report, the final gate and the runbook",
               ("S09-13", "S09-14", "S09-15", "S09-16", "S09-17", "S09-18", "S09-19",
                "S09-20", "S09-21", "S09-22", "S09-23", "S09-24", "C-10")),
        "W6": ("Gate G-1: the audit, the reproduction and the guards",
               ("S09-34", "S09-35", "S09-36", "S09-37", "S09-38", "S09-39")),
    },
}
TASK_PLANS = {"S09-W1": "docs/stage9_plan.md §10", "S09-W2": "docs/stage9_plan.md §13–§16",
              "S09-W3": "docs/stage9_plan.md §17", "S09-W4": "docs/stage9_plan.md §18",
              "S09-W5": "docs/stage9_plan.md §19", "S09-W6": "docs/stage9_plan.md §20"}
CORRECTION_TASK = "corrections"


def stage_key(stage: str) -> str:
    return f"S{stage.zfill(2)}"


def task_of(spec: FigureSpec) -> str:
    """The task whose figure sheet holds the figure: S09-W5, S06, or the figure book's
    corrections."""
    hits = [f"{stage_key(stage)}-{item}" for stage, items in WORK_ITEMS.items()
            for item, (_, ids) in items.items() if spec.id in ids]
    assert len(hits) <= 1, f"{spec.id} is listed in {hits}"
    if hits:
        return hits[0]
    if spec.kind == "correction":
        return CORRECTION_TASK
    assert spec.stage not in WORK_ITEMS, f"{spec.id}: stage {spec.stage} has work items; list it"
    return stage_key(spec.stage)


def task_title(key: str) -> str:
    if key == CORRECTION_TASK:
        return "Corrections of the submitted project made with the figure book (OP-56)"
    for stage, items in WORK_ITEMS.items():
        for item, (title, _) in items.items():
            if key == f"{stage_key(stage)}-{item}":
                return f"Stage {stage}, {item}: {title}"
    stage = next(s for s in STAGES if stage_key(s) == key)
    return f"Stage {stage}: {STAGE_NAMES[stage]}"


def tasks(specs: list[FigureSpec]) -> dict[str, list[FigureSpec]]:
    """Task key -> its figures, in stage and work-item order, process figures first."""
    out: dict[str, list[FigureSpec]] = {}
    for s in specs:
        out.setdefault(task_of(s), []).append(s)
    order = [stage_key(s) for s in STAGES]
    rank = {k: (order.index(k.split("-")[0]) if k.split("-")[0] in order else len(order), k)
            for k in out}
    return {k: sorted(v, key=lambda f: (f.kind != "process", f.id))
            for k, v in sorted(out.items(), key=lambda kv: rank[kv[0]])}


def load_all() -> list[FigureSpec]:
    """Import every stage module so that its figures register, in stage order."""
    from solarpv.figures import (  # noqa: F401
        correction,
        s00_s02,
        s04_s05,
        s06,
        s07_s09,
        s09_models,
        s09_w5,
        s09_w6,
    )
    order = {s: i for i, s in enumerate(STAGES)}
    ids = [f.id for f in REGISTRY]
    assert len(ids) == len(set(ids)), "duplicate figure id"
    return sorted(REGISTRY, key=lambda f: (order[f.stage], f.id))
