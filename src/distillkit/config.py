"""Typed configuration for every pipeline stage.

An experiment is described by one YAML file (``configs/experiments/*.yaml``) that
references a dataset, a teacher and a list of students, each in its own YAML file.
References are plain relative paths; they are resolved against the directory of the
file that contains them, then against the current working directory.

All models are Pydantic v2 models with ``extra="forbid"`` so that a typo in a config
key fails loudly instead of being silently ignored.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- dataset


class DatasetSource(StrictModel):
    """One way of obtaining the raw dataset.

    ``loader`` is either the name of a built-in loader (``huggingface``, ``csv``,
    ``jsonl``, ``clinc_json``) or an import path ``package.module:function`` for a
    custom loader. ``args`` are passed to the loader as keyword arguments.
    """

    loader: str
    args: dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(StrictModel):
    name: str
    description: str = ""
    task: str = "intent classification of short customer-support messages"
    sources: list[DatasetSource] = Field(min_length=1)
    validation_fraction: float = Field(default=0.1, gt=0.0, lt=0.5)
    seed: int = 13
    # Optional per-split caps, e.g. {"train": 200} for development slices.
    max_examples: dict[Literal["train", "validation", "test"], int] = Field(default_factory=dict)
    # Label that marks out-of-scope messages (CLINC150 "oos"); informational for reports.
    oos_label: str | None = None


# --------------------------------------------------------------------------- teacher


class TokenPricing(StrictModel):
    """USD per one million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float | None = Field(default=None, ge=0)
    cache_write: float | None = Field(default=None, ge=0)
    source: str = ""
    as_of: str = ""


class SimulatedTeacherConfig(StrictModel):
    """Settings for the ``simulated`` backend (tests and dry runs only)."""

    accuracy: float = Field(default=0.9, ge=0.0, le=1.0)
    invalid_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int = 0
    latency_ms: float = Field(default=0.0, ge=0.0)


class TeacherConfig(StrictModel):
    name: str
    backend: Literal["anthropic", "openai", "simulated"]
    model: str
    prompt: Path
    base_url: str | None = None
    # Name of the environment variable holding the API key. Keys are never stored in config.
    api_key_env: str | None = None
    max_output_tokens: int = Field(default=256, gt=0)
    # None means "do not send a temperature" (some models reject non-default sampling).
    temperature: float | None = None
    structured_output: Literal["json_schema", "json_object", "none"] = "json_schema"
    # Extra request fields merged (deep) into every request body, e.g. effort settings.
    request_options: dict[str, Any] = Field(default_factory=dict)
    # Number of samples per message; >1 enables self-consistency voting.
    self_consistency: int = Field(default=1, ge=1, le=15)
    # Runtime knobs. They do not affect labels, so they are excluded from fingerprints.
    concurrency: int = Field(default=8, ge=1)
    requests_per_minute: float | None = Field(default=None, gt=0)
    max_retries: int = Field(default=6, ge=0)
    timeout_s: float = Field(default=60.0, gt=0)
    max_budget_usd: float | None = Field(default=None, gt=0)
    pricing: TokenPricing | None = None
    is_local: bool = False
    simulated: SimulatedTeacherConfig | None = None

    RUNTIME_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"concurrency", "requests_per_minute", "max_retries", "timeout_s", "max_budget_usd"}
    )

    def identity(self) -> dict[str, Any]:
        """Fields that determine the labels the teacher produces (runtime knobs excluded)."""
        data = self.model_dump(mode="json", exclude=set(self.RUNTIME_FIELDS))
        # The prompt is identified by its content, not by where it lives on disk.
        data.pop("prompt", None)
        return data


# --------------------------------------------------------------------------- students


class _StudentBase(StrictModel):
    name: str
    # Train on the teacher's vote distribution (soft labels) instead of the majority label.
    # Only has an effect when the teacher was sampled more than once.
    use_soft_labels: bool = False


class TfidfParams(StrictModel):
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] | None = (2, 5)
    min_df: int = 1
    max_features: int | None = 200_000
    sublinear_tf: bool = True
    C: float = Field(default=10.0, gt=0)
    max_iter: int = 2000


class TfidfStudentConfig(_StudentBase):
    kind: Literal["tfidf"] = "tfidf"
    params: TfidfParams = Field(default_factory=TfidfParams)


class SetFitParams(StrictModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 64
    # Contrastive fine-tuning of the embedding body (the "Set" in SetFit). 0 disables it.
    contrastive_epochs: int = Field(default=1, ge=0)
    max_pairs: int = Field(default=20_000, ge=0)
    contrastive_batch_size: int = 32
    learning_rate: float = 2e-5
    embed_batch_size: int = 128
    head_C: float = Field(default=10.0, gt=0)
    head_max_iter: int = 2000
    normalize_embeddings: bool = True


class SetFitStudentConfig(_StudentBase):
    kind: Literal["setfit"] = "setfit"
    params: SetFitParams = Field(default_factory=SetFitParams)


class EncoderParams(StrictModel):
    model_name: str = "distilbert-base-uncased"
    max_length: int = 64
    epochs: int = Field(default=5, ge=1)
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = Field(default=0.1, ge=0, lt=1)
    # Distillation loss: alpha * T^2 * KL(teacher || student) + (1 - alpha) * CE(hard).
    kd_alpha: float = Field(default=0.5, ge=0, le=1)
    kd_temperature: float = Field(default=2.0, gt=0)
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)
    early_stopping_patience: int = Field(default=2, ge=1)
    fp16: bool = True  # only used on CUDA


class EncoderStudentConfig(_StudentBase):
    kind: Literal["encoder"] = "encoder"
    params: EncoderParams = Field(default_factory=EncoderParams)


StudentConfig = Annotated[
    TfidfStudentConfig | SetFitStudentConfig | EncoderStudentConfig,
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------- experiment


class LabelFilterConfig(StrictModel):
    """Drop teacher labels whose self-consistency agreement is below a threshold."""

    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class TrackingConfig(StrictModel):
    backend: Literal["mlflow", "none"] = "none"
    tracking_uri: str = "sqlite:///mlruns.db"
    experiment_name: str | None = None


class CpuCostConfig(StrictModel):
    """Reference CPU instance used to turn measured throughput into dollars."""

    instance: str = "AWS c7i.xlarge (4 vCPU, on-demand, us-east-1)"
    usd_per_hour: float = Field(default=0.1785, gt=0)
    price_as_of: str = "unverified"
    source: str = "https://aws.amazon.com/ec2/pricing/on-demand/"


class LatencyConfig(StrictModel):
    warmup: int = Field(default=20, ge=0)
    requests: int = Field(default=300, ge=10)
    throughput_batch_size: int = Field(default=64, ge=1)
    throughput_seconds: float = Field(default=5.0, gt=0)
    threads: int | None = 4


class RoutingConfig(StrictModel):
    thresholds: list[float] = Field(
        default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    )


class BenchmarkConfig(StrictModel):
    cpu: CpuCostConfig = Field(default_factory=CpuCostConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    ece_bins: int = Field(default=15, ge=2)
    confused_pairs: int = Field(default=10, ge=1)
    examples_per_pair: int = Field(default=3, ge=0)


class ExportConfig(StrictModel):
    # "best" picks the full-budget student with the highest mean validation accuracy
    # (validation labels come from the teacher; the test set is never used for selection).
    student: str = "best"
    quantize: bool = True
    opset: int = 17


Budget = int | Literal["full"]


class ExperimentConfig(StrictModel):
    name: str
    dataset: DatasetConfig
    teacher: TeacherConfig
    students: list[StudentConfig] = Field(min_length=1)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], min_length=1)
    budgets: list[Budget] = Field(default_factory=lambda: list[Budget](["full"]), min_length=1)
    label_filter: LabelFilterConfig = Field(default_factory=LabelFilterConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    artifacts_dir: Path = Path("artifacts")
    results_dir: Path = Path("results")

    @field_validator("students")
    @classmethod
    def _unique_student_names(cls, students: list[StudentConfig]) -> list[StudentConfig]:
        names = [s.name for s in students]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate student names: {duplicates}")
        return students

    @field_validator("budgets")
    @classmethod
    def _positive_budgets(cls, budgets: list[Budget]) -> list[Budget]:
        for b in budgets:
            if isinstance(b, int) and b <= 0:
                raise ValueError(f"budgets must be positive, got {b}")
        return budgets

    def student(self, name: str) -> TfidfStudentConfig | SetFitStudentConfig | EncoderStudentConfig:
        for s in self.students:
            if s.name == name:
                return s
        raise KeyError(f"unknown student {name!r}; known: {[s.name for s in self.students]}")

    @property
    def experiment_dir(self) -> Path:
        return self.artifacts_dir / self.name

    @property
    def experiment_results_dir(self) -> Path:
        return self.results_dir / self.name


# --------------------------------------------------------------------------- loading

# Keys whose string values are references to other YAML files.
_REFERENCE_KEYS = ("dataset", "teacher", "students")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def _resolve_path(ref: str | Path, base_dir: Path) -> Path:
    candidate = Path(ref)
    if candidate.is_absolute():
        return candidate
    for root in (base_dir, Path.cwd()):
        if (root / candidate).exists():
            return root / candidate
    return base_dir / candidate


def _load_reference(value: Any, base_dir: Path) -> Any:
    """Inline a YAML file referenced by path; relative ``prompt`` paths follow the file."""
    if isinstance(value, str) and value.endswith((".yaml", ".yml")):
        path = _resolve_path(value, base_dir)
        data = _read_yaml(path)
        if "prompt" in data and isinstance(data["prompt"], str):
            data["prompt"] = str(_resolve_path(data["prompt"], path.parent))
        return data
    return value


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides; values are parsed as YAML scalars/collections.

    List items can be addressed by index or, for lists of named mappings such as
    ``students``, by name: ``students.tfidf-logreg.params.C=3``.
    """
    data = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key.path=value")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        parts = key.strip().split(".")
        node: Any = data
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            if isinstance(node, list):
                idx = _list_index(node, part)
                if last:
                    node[idx] = value
                else:
                    node = node[idx]
            elif isinstance(node, dict):
                if last:
                    node[part] = value
                else:
                    node = node.setdefault(part, {})
            else:
                raise ValueError(f"cannot apply override {item!r}: {part!r} is not a container")
    return data


def _list_index(node: list[Any], part: str) -> int:
    if part.isdigit():
        return int(part)
    for i, element in enumerate(node):
        if isinstance(element, Mapping) and element.get("name") == part:
            return i
    raise KeyError(f"no list element named {part!r}")


def _inline_references(data: dict[str, Any], base: Path) -> dict[str, Any]:
    for key in _REFERENCE_KEYS:
        if key not in data:
            continue
        if key == "students":
            data[key] = [_load_reference(v, base) for v in data[key]]
        else:
            data[key] = _load_reference(data[key], base)
    return data


def load_experiment_dict(path: str | Path) -> dict[str, Any]:
    """Read an experiment YAML file and inline every referenced file."""
    path = Path(path)
    return _inline_references(_read_yaml(path), path.parent)


def load_experiment(path: str | Path, overrides: Iterable[str] = ()) -> ExperimentConfig:
    """Load an experiment config, apply overrides and validate it.

    An override may also swap a whole referenced file, e.g.
    ``teacher=configs/teachers/simulated.yaml``.
    """
    path = Path(path)
    data = apply_overrides(load_experiment_dict(path), overrides)
    data = _inline_references(data, path.parent)
    return ExperimentConfig.model_validate(data)


def load_teacher(path: str | Path, overrides: Iterable[str] = ()) -> TeacherConfig:
    path = Path(path)
    data = _load_reference(str(path), path.parent)
    return TeacherConfig.model_validate(apply_overrides(data, overrides))


def load_pricing_table(path: str | Path | None = None) -> dict[str, TokenPricing]:
    """Load the published-price table shipped in ``configs/pricing.yaml``."""
    path = Path(path) if path else _default_pricing_path()
    data = _read_yaml(path)
    meta = {k: data.get(k, "") for k in ("as_of", "source")}
    table: dict[str, TokenPricing] = {}
    for model, entry in (data.get("models") or {}).items():
        merged = {**meta, **entry}
        table[model] = TokenPricing.model_validate(merged)
    return table


def _default_pricing_path() -> Path:
    for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
        candidate = root / "configs" / "pricing.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("configs/pricing.yaml not found; pass a pricing path explicitly")
