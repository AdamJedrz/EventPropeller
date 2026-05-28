from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class PipelineConfig:
    n_propellers: int = 2
    chunk_size: int = 5_000_000
    bundle_us: int = 2_000
    bundles_per_window: int = 10
    polarity_mode: str = "positive"
    start_time_ms: float = 0.0
    max_windows: Optional[int] = None
    bin_size: int = 7
    min_events_in_bin: int = 3
    density_min_component_area: int = 18
    remove_border_components: bool = True
    density_bin_morph_mode: str = "none"
    density_bin_morph_kernel_size: int = 3
    density_bin_morph_iterations: int = 0
    cluster_bbox_pad: int = 12
    cluster_min_events_per_cluster: int = 40
    cluster_sample_max_events: int = 12000
    cluster_attempts: int = 5
    bootstrap_sort: str = "x"
    max_centroid_dx_per_window: Optional[float] = 20.0
    max_centroid_dy_per_window: Optional[float] = 20.0
    max_centroid_distance_per_window: Optional[float] = None
    rpm_abs_max: float = 8000
    rpm_step_coarse: float = 200
    rpm_step_fine: float = 20
    rpm_refine_span: float = 200
    rpm_local_delta: float = 150
    full_search_period_us: int = 500_000
    min_events_for_rpm: int = 50
    max_events_mc: int = 3000
    downsample_time_bins: int = 20
    parallel_mc: bool = False
    parallel_candidate_chunk_size: int = 512
    score_mode: str = "mean_square"
    mc_score_lambda: float = 1.0
    mc_score_eps: float = 1e-6
    reference_time_fractions: Tuple[float, ...] = field(default_factory=lambda: (0.5,))
    center_search_radius_px: float = 0.0
    center_search_step_px: float = 1.0
    q_search_enabled: bool = False
    q_fixed: float = 1.0
    q_min: float = 0.50
    q_max: float = 1.00
    q_step: float = 0.05
    q_axis_angle_deg: float = 0.0
    q_axis_search_enabled: bool = False
    q_axis_angle_min_deg: float = 0.0
    q_axis_angle_max_deg: float = 165.0
    q_axis_angle_step_deg: float = 15.0
    preview: bool = False
    preview_wait_ms: int = 1
    preview_every_n_windows: int = 1
    verbose_chunks: bool = False
