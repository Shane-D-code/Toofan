"""
losses.py
---------
Haversine great-circle distance (differentiable, in torch) and a
physics-aware, uncertainty-weighted loss for multi-horizon trajectory
prediction.

V2 changes:

  * The Gaussian NLL is now computed entirely in the same coordinate
    space the model predicts uncertainty in: raw-degree (dlat, dlon)
    deltas. V1 combined a Haversine distance in km^2 with a log-variance
    defined in degree/normalized space, which is not mathematically
    consistent (the "precision" term did not have the right units to
    scale a km^2 error). Haversine km is still computed and used for the
    primary trajectory loss and for all reported evaluation metrics --
    it's simply no longer inside the NLL term.
  * The physics speed penalty now measures actual predicted cyclone
    translation speed: Haversine displacement from the CURRENT position
    to the PREDICTED future position, divided by the forecast horizon.
    V1 computed "distance between prediction and truth / horizon", which
    is a measure of prediction error growth, not of how fast the model
    thinks the storm is moving, and would have penalized accurate
    predictions of genuinely fast-moving storms.
  * Multi-horizon losses are computed explicitly per horizon and combined
    with configurable weights, so the (typically easiest) shortest
    horizon cannot dominate the gradient.
  * `MAX_PLAUSIBLE_SPEED_KMH` and the per-horizon weights are parameters
    of `physics_aware_loss`, not hard-coded module constants, per the
    "no hard-coded experimental values" requirement.
"""
import torch

EARTH_RADIUS_KM = 6371.0088
DEFAULT_MAX_PLAUSIBLE_SPEED_KMH = 90.0  # ~fastest observed TC translation speeds
DEFAULT_LOG_VAR_MIN = -10.0
DEFAULT_LOG_VAR_MAX = 5.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. All inputs in degrees, any matching shape."""
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    a = torch.clamp(a, 0.0, 1.0)
    return 2 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(a))


def physics_aware_loss(pred, target_delta, current_pos, horizon_hours,
                        horizon_weights=None,
                        max_plausible_speed_kmh=DEFAULT_MAX_PLAUSIBLE_SPEED_KMH,
                        nll_weight=0.1, physics_weight=0.05,
                        log_var_min=DEFAULT_LOG_VAR_MIN, log_var_max=DEFAULT_LOG_VAR_MAX,
                        eps=1e-6):
    """
    pred            : dict from CycloneTransformer -> {"delta": (B,H,2), "log_var": (B,H,2)}
                       both in raw-degree (dlat, dlon) space.
    target_delta    : (B, H, 2) true (dlat, dlon) in degrees, for each horizon
    current_pos     : (B, 2) current (lat, lon) in degrees -- the anchor point
    horizon_hours   : (H,) tensor/list, actual forecast lead time in hours for each horizon
    horizon_weights : (H,) tensor/list, per-horizon weights for the trajectory loss.
                       Defaults to uniform weights if not given.

    Loss = weighted-mean-over-horizons Haversine trajectory loss (primary)
           + nll_weight   * Gaussian NLL in degree space (uncertainty calibration)
           + physics_weight * speed-plausibility penalty (soft physics regularizer)

    Returns: total loss (scalar), and a dict of components for logging
             (including per-horizon Haversine km, keyed by the horizon's
             hour value, for per-horizon validation reporting).
    """
    delta = pred["delta"]        # (B, H, 2) degrees
    log_var = torch.clamp(pred["log_var"], min=log_var_min, max=log_var_max)  # (B, H, 2)

    horizon_hours_t = torch.as_tensor(horizon_hours, dtype=delta.dtype, device=delta.device)
    H = delta.shape[1]
    if horizon_weights is None:
        horizon_weights = [1.0] * H
    w = torch.as_tensor(horizon_weights, dtype=delta.dtype, device=delta.device)
    assert w.shape[0] == H, f"horizon_weights length {w.shape[0]} != num horizons {H}"

    pred_lat = current_pos[:, 0:1] + delta[..., 0]
    pred_lon = current_pos[:, 1:2] + delta[..., 1]
    true_lat = current_pos[:, 0:1] + target_delta[..., 0]
    true_lon = current_pos[:, 1:2] + target_delta[..., 1]

    # --- Primary trajectory loss: Haversine displacement error per horizon (km) ---
    dist_km = haversine_km(pred_lat, pred_lon, true_lat, true_lon)  # (B, H)
    per_horizon_km = dist_km.mean(dim=0)  # (H,) mean over batch, one value per horizon
    trajectory_loss = (per_horizon_km * w).sum() / w.sum()

    # --- Heteroscedastic NLL, computed entirely in degree space (consistent
    #     with the space `log_var` is predicted in) -- kept separate from the
    #     physical-km trajectory loss above so units are never mixed. ---
    sq_err = (delta - target_delta) ** 2               # (B, H, 2), degrees^2
    var = torch.exp(log_var) + eps
    nll = 0.5 * (sq_err / var) + 0.5 * log_var
    nll = nll.mean()

    # --- Physics-aware regularizer: penalize implausible PREDICTED translation
    #     speed (current position -> predicted position), not prediction error. ---
    pred_disp_km = haversine_km(
        current_pos[:, 0:1].expand_as(pred_lat), current_pos[:, 1:2].expand_as(pred_lon),
        pred_lat, pred_lon,
    )  # (B, H)
    predicted_speed = pred_disp_km / horizon_hours_t.unsqueeze(0)  # (B, H) km/h
    speed_violation = torch.relu(predicted_speed - max_plausible_speed_kmh)
    physics_penalty = (speed_violation ** 2).mean()

    total = trajectory_loss + nll_weight * nll + physics_weight * physics_penalty

    components = {
        "loss_total": total.item(),
        "loss_trajectory": trajectory_loss.item(),
        "loss_nll": nll.item(),
        "loss_physics": physics_penalty.item(),
        "mean_haversine_km": dist_km.mean().item(),
        "per_horizon_km": {float(h): float(per_horizon_km[i].item()) for i, h in enumerate(horizon_hours)},
    }
    return total, components


# ---------------------------------------------------------------------------
# V3 loss: long-horizon-aware weighted position loss + trajectory/speed
# consistency + acceleration/turning regularization.
#
# The PRIMARY objective remains accurate future-position prediction (the
# weighted per-horizon Haversine term). The consistency / speed / acceleration
# / turning terms are small regularizers that only penalize EXTREME
# discontinuities in the predicted trajectory, so cyclones are still free to
# accelerate, decelerate and curve naturally.
# ---------------------------------------------------------------------------
# Default thresholds / weights -- all configurable at the call site (and
# injected through train.py CLI args), NEVER hard-coded assumptions baked in.
DEFAULT_TRAJECTORY_CONSISTENCY_WEIGHT = 0.05
DEFAULT_SPEED_CONSISTENCY_WEIGHT = 0.05
DEFAULT_ACCELERATION_WEIGHT = 0.02
DEFAULT_TURNING_WEIGHT = 0.02
DEFAULT_MAX_SEGMENT_SPEED_CHANGE_KMH = 40.0   # extreme speed change between segments (km/h)
DEFAULT_MAX_TURN_DEG = 90.0                    # extreme heading change between segments (deg)


def _segment_speeds(pred_pos, current_pos, horizon_hours_t):
    """Per-segment predicted translation speed (km/h) between consecutive
    forecast points, using ACTUAL interval durations. Segment h goes from
    pred point h-1 (or the current position for h==0) to pred point h, over
    duration (horizon_h - horizon_{h-1})."""
    prev_pos = torch.cat([current_pos[:, None, :], pred_pos[:, :-1, :]], dim=1)  # (B,H,2)
    seg_km = haversine_km(prev_pos[..., 0], prev_pos[..., 1], pred_pos[..., 0], pred_pos[..., 1])  # (B,H)
    dur = horizon_hours_t.clone()
    prev_h = torch.cat([torch.zeros(1, device=dur.device, dtype=dur.dtype), dur[:-1]])
    seg_dur = dur - prev_h  # (H,)
    seg_speed = seg_km / seg_dur.clamp_min(1e-3).unsqueeze(0)  # (B,H) km/h
    return seg_speed


def _segment_angle_deg(pred_pos, current_pos):
    """Heading (initial great-circle bearing, degrees) of each predicted
    trajectory segment (current->3h, 3h->6h, 6h->12h, 12h->24h)."""
    prev_pos = torch.cat([current_pos[:, None, :], pred_pos[:, :-1, :]], dim=1)  # (B,H,2)
    lat1, lon1 = torch.deg2rad(prev_pos[..., 0]), torch.deg2rad(prev_pos[..., 1])
    lat2, lon2 = torch.deg2rad(pred_pos[..., 0]), torch.deg2rad(pred_pos[..., 1])
    dlon = lon2 - lon1
    x = torch.sin(dlon) * torch.cos(lat2)
    y = (torch.cos(lat1) * torch.sin(lat2)
         - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon))
    brng_rad = torch.atan2(x, y)
    return torch.rad2deg(brng_rad)  # (B,H)


def cycloned_v3_loss(pred, target_delta, current_pos, horizon_hours,
                     horizon_weights=None,
                     nll_weight=0.1,
                     max_plausible_speed_kmh=DEFAULT_MAX_PLAUSIBLE_SPEED_KMH,
                     trajectory_consistency_weight=DEFAULT_TRAJECTORY_CONSISTENCY_WEIGHT,
                     speed_consistency_weight=DEFAULT_SPEED_CONSISTENCY_WEIGHT,
                     acceleration_weight=DEFAULT_ACCELERATION_WEIGHT,
                     turning_weight=DEFAULT_TURNING_WEIGHT,
                     max_segment_speed_change_kmh=DEFAULT_MAX_SEGMENT_SPEED_CHANGE_KMH,
                     max_turn_deg=DEFAULT_MAX_TURN_DEG,
                     log_var_min=DEFAULT_LOG_VAR_MIN, log_var_max=DEFAULT_LOG_VAR_MAX,
                     eps=1e-6):
    """
    V3 loss.

    pred            : dict -> {"delta": (B,H,2), "log_var": (B,H,2)} in raw-degree space
    target_delta    : (B,H,2) true (dlat, dlon) degrees
    current_pos     : (B,2) current (lat, lon)
    horizon_hours   : (H,) actual forecast lead hours
    horizon_weights : (H,) per-horizon weights for the primary position loss,
                      ordered [3h, 6h, 12h, 24h]. Defaults to uniform.

    TOTAL = weighted position loss
          + nll_weight        * Gaussian NLL (degree-space uncertainty term)
          + speed_consistency_weight    * exceedance of max per-segment speed (absurd jumps)
          + trajectory_consistency_weight * exceedance of max per-segment speed CHANGE
          + acceleration_weight  * magnitude of velocity-vector change between segments
          + turning_weight       * exceedance of max heading change between segments

    All consistency regularizers are hinge-style (0 below threshold) so they
    never penalize realistic/curving cyclone motion -- only extreme
    discontinuities. Returns (total, components) with per-horizon metrics.
    """
    delta = pred["delta"]                                # (B,H,2) degrees
    log_var = torch.clamp(pred["log_var"], min=log_var_min, max=log_var_max)

    horizon_hours_t = torch.as_tensor(horizon_hours, dtype=delta.dtype, device=delta.device)
    H = delta.shape[1]
    if horizon_weights is None:
        horizon_weights = [1.0] * H
    w = torch.as_tensor(horizon_weights, dtype=delta.dtype, device=delta.device)
    assert w.shape[0] == H, f"horizon_weights length {w.shape[0]} != num horizons {H}"

    # Predicted absolute positions (P_h = current + delta_h)
    pred_lat = current_pos[:, 0:1] + delta[..., 0]       # (B,H)
    pred_lon = current_pos[:, 1:2] + delta[..., 1]
    pred_pos = torch.stack([pred_lat, pred_lon], dim=-1) # (B,H,2)
    true_lat = current_pos[:, 0:1] + target_delta[..., 0]
    true_lon = current_pos[:, 1:2] + target_delta[..., 1]

    # --- 1) PRIMARY: weighted per-horizon Haversine position loss (dominant) ---
    dist_km = haversine_km(pred_lat, pred_lon, true_lat, true_lon)            # (B,H)
    per_horizon_km = dist_km.mean(dim=0)                                      # (H,)
    trajectory_loss = (per_horizon_km * w).sum() / w.sum()

    # --- 2) Heteroscedastic NLL in degree space (uncertainty calibration) ---
    sq_err = (delta - target_delta) ** 2
    var = torch.exp(log_var) + eps
    nll = (0.5 * (sq_err / var) + 0.5 * log_var).mean()

    # --- 3) Speed consistency / displacement consistency (#8 & #9) ---
    # Per-segment predicted speed; penalize ONLY implausibly fast segments.
    seg_speed = _segment_speeds(pred_pos, current_pos, horizon_hours_t)       # (B,H)
    speed_viol = torch.relu(seg_speed - max_plausible_speed_kmh)
    speed_penalty = (speed_viol ** 2).mean()

    # --- 4) Trajectory consistency (#7): penalize extreme SUDDEN changes of
    #       translation speed between consecutive segments (i.e. large
    #       acceleration in speed terms). Cyclones may accelerate/decelerate,
    #       so only jumps beyond max_segment_speed_change_kmh are penalized.
    if H >= 2:
        dspeed = (seg_speed[:, 1:] - seg_speed[:, :-1]).abs()                 # (B,H-1)
        traj_viol = torch.relu(dspeed - max_segment_speed_change_kmh)
        traj_penalty = (traj_viol ** 2).mean()
    else:
        traj_penalty = torch.zeros((), device=delta.device)

    # --- 5) Acceleration regularization (#10): magnitude of the CHANGE in the
    #       east/north velocity vector between consecutive segments (vector
    #       acceleration), penalized only when extreme.
    if H >= 3:
        # approximate east/north displacement vectors per segment (degrees * local scale)
        u_seg = 111.32 * pred_lon.diff(dim=1, prepend=current_pos[:, 1:2].expand(-1, 1)) \
            * torch.cos(torch.deg2rad(pred_lat)).clamp_min(0.1)
        v_seg = 111.32 * pred_lat.diff(dim=1, prepend=current_pos[:, 0:1].expand(-1, 1))
        # velocity per segment over its duration
        dur = horizon_hours_t.diff(prepend=torch.zeros(1, dtype=horizon_hours_t.dtype,
                                                       device=horizon_hours_t.device))
        u_seg_v = u_seg / dur.clamp_min(1e-3).unsqueeze(0)   # km/h
        v_seg_v = v_seg / dur.clamp_min(1e-3).unsqueeze(0)
        dvec = torch.sqrt((u_seg_v[:, 1:] - u_seg_v[:, :-1]) ** 2
                          + (v_seg_v[:, 1:] - v_seg_v[:, :-1]) ** 2)          # km/h
        accel_viol = torch.relu(dvec - max_segment_speed_change_kmh)
        accel_penalty = (accel_viol ** 2).mean()
    else:
        accel_penalty = torch.zeros((), device=delta.device)

    # --- 6) Turning regularization (#10): extreme heading change between
    #       consecutive predicted segments; cyclones curve, so only penalize
    #       unrealistically abrupt direction flips.
    if H >= 2:
        heading = _segment_angle_deg(pred_pos, current_pos)                    # (B,H) deg
        dheading = (heading[:, 1:] - heading[:, :-1]).abs()
        dheading = torch.abs((dheading + 180.0) % 360.0 - 180.0)               # wrap to [-180,180]
        turn_viol = torch.relu(dheading - max_turn_deg)
        turn_penalty = (turn_viol ** 2).mean()
    else:
        turn_penalty = torch.zeros((), device=delta.device)

    total = (trajectory_loss
             + nll_weight * nll
             + speed_consistency_weight * speed_penalty
             + trajectory_consistency_weight * traj_penalty
             + acceleration_weight * accel_penalty
             + turning_weight * turn_penalty)

    components = {
        "loss_total": total.item(),
        "loss_trajectory": trajectory_loss.item(),
        "loss_nll": nll.item(),
        "loss_speed_consistency": speed_penalty.item(),
        "loss_trajectory_consistency": traj_penalty.item(),
        "loss_acceleration": accel_penalty.item(),
        "loss_turning": turn_penalty.item(),
        "mean_haversine_km": dist_km.mean().item(),
        "per_horizon_km": {float(h): float(per_horizon_km[i].item()) for i, h in enumerate(horizon_hours)},
    }
    return total, components


# ---------------------------------------------------------------------------
# V5 loss: target-representation-aware and mode-independent.
#
# The model's `delta` (and the dataset target) live in one of:
#   "delta"  (deg) | "km" (local km) | "motion" (segment velocity km/h)
# This loss first reconstructs BOTH predicted and true absolute (lat, lon)
# positions (via dataset.reconstruct_positions), then evaluates everything in
# physical great-circle km. So the primary term is a uniform Haversine loss
# regardless of representation, and all physics/trajectory-consistency terms are
# computed on physical positions/speeds (mode-independent).
#
# Components (all configurable via weights/masks for ablations, Experiment C/D):
#   * primary weighted-per-horizon Haversine position loss (interval 0..1 on)
#   * heteroscedastic NLL in the model's own output space (nll_weight)
#   * speed-consistency: exceedance of implausible per-segment translation speed
#   * trajectory-consistency: exceedance of extreme per-segment speed CHANGE
#   * turning: exceedance of abrupt heading change between segments
#   * acceleration: magnitude of velocity-vector change between segments
#   * direction-persistence: consistency of consecutive segment headings
# ---------------------------------------------------------------------------
def v5_loss(pred, target, cur_pos, horizons, target_mode="delta",
            horizon_weights=None,
            nll_weight=0.1,
            max_plausible_speed_kmh=DEFAULT_MAX_PLAUSIBLE_SPEED_KMH,
            speed_consistency_weight=0.05,
            trajectory_consistency_weight=0.05,
            acceleration_weight=0.02,
            turning_weight=0.02,
            direction_weight=0.0,
            max_segment_speed_change_kmh=DEFAULT_MAX_SEGMENT_SPEED_CHANGE_KMH,
            max_turn_deg=DEFAULT_MAX_TURN_DEG,
            log_var_min=DEFAULT_LOG_VAR_MIN, log_var_max=DEFAULT_LOG_VAR_MAX,
            eps=1e-6):
    from dataset import reconstruct_positions
    delta = pred["delta"]
    log_var = torch.clamp(pred["log_var"], min=log_var_min, max=log_var_max)
    H = delta.shape[1]
    horizons_t = torch.as_tensor(horizons, dtype=delta.dtype, device=delta.device)
    if horizon_weights is None:
        horizon_weights = [1.0] * H
    w = torch.as_tensor(horizon_weights, dtype=delta.dtype, device=delta.device)
    assert w.shape[0] == H

    pred_pos = reconstruct_positions(delta, cur_pos, horizons, target_mode)   # (B,H,2)
    true_pos = reconstruct_positions(target, cur_pos, horizons, target_mode)  # (B,H,2)

    # --- 1) PRIMARY: weighted per-horizon Haversine position loss ---
    dist_km = haversine_km(pred_pos[..., 0], pred_pos[..., 1],
                           true_pos[..., 0], true_pos[..., 1])                # (B,H)
    per_horizon_km = dist_km.mean(dim=0)
    trajectory_loss = (per_horizon_km * w).sum() / w.sum()

    # --- 2) Heteroscedastic NLL in the model's output (target) space ---
    sq_err = (delta - target) ** 2
    var = torch.exp(log_var) + eps
    nll = (0.5 * (sq_err / var) + 0.5 * log_var).mean()

    # --- physics / trajectory consistency in physical units ---
    prev_pos = torch.cat([cur_pos[:, None, :], pred_pos[:, :-1, :]], dim=1)   # (B,H,2)
    seg_km = haversine_km(prev_pos[..., 0], prev_pos[..., 1],
                          pred_pos[..., 0], pred_pos[..., 1])                 # (B,H)
    prev_h = torch.cat([torch.zeros(1, device=horizons_t.device, dtype=horizons_t.dtype),
                        horizons_t[:-1]])
    seg_dur = (horizons_t - prev_h).clamp_min(1e-3).unsqueeze(0)              # (1,H)
    seg_speed = seg_km / seg_dur                                              # (B,H) km/h

    speed_viol = torch.relu(seg_speed - max_plausible_speed_kmh)
    speed_penalty = (speed_viol ** 2).mean()

    if H >= 2:
        dspeed = (seg_speed[:, 1:] - seg_speed[:, :-1]).abs()
        traj_penalty = (torch.relu(dspeed - max_segment_speed_change_kmh) ** 2).mean()
    else:
        traj_penalty = torch.zeros((), device=delta.device)

    s_prev = None
    accel_penalty = torch.zeros((), device=delta.device)
    if H >= 3:
        u_seg = 111.32 * pred_pos[..., 1].diff(dim=1, prepend=cur_pos[:, 1:2].expand(-1, 1)) \
            * torch.cos(torch.deg2rad(pred_pos[..., 0])).clamp_min(0.1)
        v_seg = 111.32 * pred_pos[..., 0].diff(dim=1, prepend=cur_pos[:, 0:1].expand(-1, 1))
        dur = horizons_t.diff(prepend=torch.zeros(1, dtype=horizons_t.dtype, device=horizons_t.device))
        u_seg_v = u_seg / dur.clamp_min(1e-3).unsqueeze(0)
        v_seg_v = v_seg / dur.clamp_min(1e-3).unsqueeze(0)
        dvec = torch.sqrt((u_seg_v[:, 1:] - u_seg_v[:, :-1]) ** 2
                          + (v_seg_v[:, 1:] - v_seg_v[:, :-1]) ** 2)
        accel_penalty = (torch.relu(dvec - max_segment_speed_change_kmh) ** 2).mean()

    heading = _segment_angle_deg(pred_pos, cur_pos)                           # (B,H) deg
    turn_penalty = torch.zeros((), device=delta.device)
    if H >= 2:
        dheading = (heading[:, 1:] - heading[:, :-1]).abs()
        dheading = torch.abs((dheading + 180.0) % 360.0 - 180.0)
        turn_penalty = (torch.relu(dheading - max_turn_deg) ** 2).mean()

    direction_penalty = torch.zeros((), device=delta.device)
    if direction_weight > 0 and H >= 2:
        # Direction persistence: penalize heading flips between successive
        # segments that are large (here > 90 deg), weighted by the geometric
        # mean of the two segment speeds so a reversal of a fast-moving storm
        # is penalized more than a stray fix on a slow storm.
        dheading_all = torch.abs((heading[:, 1:] - heading[:, :-1]).abs())
        dheading_all = torch.abs((dheading_all + 180.0) % 360.0 - 180.0)
        rev = torch.relu(dheading_all - 90.0)
        speed_scale = (seg_speed[:, 1:] * seg_speed[:, :-1]).clamp_min(0.0).sqrt()
        direction_penalty = ((rev * speed_scale) ** 2).mean() * 1e-3

    total = (trajectory_loss
             + nll_weight * nll
             + speed_consistency_weight * speed_penalty
             + trajectory_consistency_weight * traj_penalty
             + acceleration_weight * accel_penalty
             + turning_weight * turn_penalty
             + direction_weight * direction_penalty)

    components = {
        "loss_total": total.item(),
        "loss_trajectory": trajectory_loss.item(),
        "loss_nll": nll.item(),
        "loss_speed_consistency": speed_penalty.item(),
        "loss_trajectory_consistency": traj_penalty.item(),
        "loss_acceleration": accel_penalty.item(),
        "loss_turning": turn_penalty.item(),
        "loss_direction": direction_penalty.item(),
        "mean_haversine_km": dist_km.mean().item(),
        "per_horizon_km": {float(h): float(per_horizon_km[i].item()) for i, h in enumerate(horizons)},
    }
    return total, components


# ---------------------------------------------------------------------------
# V6 loss: robust normalized position loss in km space.
#
# The V6 model predicts CUMULATIVE km displacement per horizon:
#   delta[h] = (dx_km EAST, dy_km NORTH)
# The PRIMARY objective is trajectory position accuracy. We use a Huber /
# SmoothL1 robust regression loss on the (dx_km, dy_km) errors for every
# horizon, NORMALIZED by a per-horizon displacement scale (so a 24h horizon is
# not dominated merely because its displacement magnitude is larger). Huber
# keeps large single-outlier errors from producing huge gradients.
#
# Optional auxiliary physics regularizers are kept near zero (spec #12) so they
# never dominate the position loss:
#   trajectory_consistency_weight = 0.01
#   speed_consistency_weight      = 0.005
#   acceleration_weight           = 0.0
#   turning_weight                = 0.0
# ---------------------------------------------------------------------------
V6_DEFAULT_HORIZON_WEIGHTS = [1.0, 1.0, 1.5, 2.0]  # [3h, 6h, 12h, 24h]


def _huber_loss(err, delta=1.0):
    """SmoothL1/Huber on a (already-scaled) error tensor, reduced to a scalar mean."""
    return torch.nn.functional.huber_loss(err, torch.zeros_like(err), delta=delta)


def v6_loss(pred, target, cur_pos, horizons, target_mode="km",
            horizon_weights=None, horizon_scales=None, huber_delta=1.0,
            trajectory_consistency_weight=0.01, speed_consistency_weight=0.005,
            acceleration_weight=0.0, turning_weight=0.0,
            max_plausible_speed_kmh=DEFAULT_MAX_PLAUSIBLE_SPEED_KMH,
            max_segment_speed_change_kmh=DEFAULT_MAX_SEGMENT_SPEED_CHANGE_KMH,
            max_turn_deg=DEFAULT_MAX_TURN_DEG, eps=1e-6):
    """
    pred     : dict -> {"delta": (B,H,2)} in the active target space.
    target   : (B,H,2) true target in the same space ("km": dx_km EAST, dy_km NORTH).
    cur_pos  : (B,2) anchor (lat, lon) degrees (needed for the geographic aux terms + metrics).
    horizons : (H,) actual forecast lead hours.
    horizon_weights : (H,) moderate per-horizon weights, default [1,1,1.5,2].
    horizon_scales  : (H,) per-horizon displacement scale (km) from the training set,
                      used to normalize the robust loss so 24h does not dominate
                      by magnitude. Defaults to 1.0 (no normalization) if not given.

    TOTAL = robust normalized per-horizon position loss
          + trajectory_consistency_weight * extreme per-segment speed-change exceedance
          + speed_consistency_weight      * extreme per-segment speed exceedance
          + acceleration_weight           * velocity-vector-change (default 0)
          + turning_weight                * extreme heading-change (default 0)

    Returns (total, components) including the physical per-horizon Haversine km
    (the reported evaluation metric) and reconstruction diagnostics.
    """
    from dataset import reconstruct_positions
    delta = pred["delta"]                    # (B,H,2) in target space
    H = delta.shape[1]
    horizons_t = torch.as_tensor(horizons, dtype=delta.dtype, device=delta.device)
    if horizon_weights is None:
        horizon_weights = V6_DEFAULT_HORIZON_WEIGHTS[:H]
    w = torch.as_tensor(horizon_weights, dtype=delta.dtype, device=delta.device)
    assert w.shape[0] == H
    if horizon_scales is None:
        scale = torch.ones(H, dtype=delta.dtype, device=delta.device)
    else:
        scale = torch.as_tensor(horizon_scales, dtype=delta.dtype, device=delta.device)
        assert scale.shape[0] == H

    # --- 1) PRIMARY: robust normalized per-horizon position loss on (dx,dy) km ---
    err = delta - target                            # (B,H,2) km
    err_n = err / scale.view(1, H, 1).clamp_min(eps)  # normalize per horizon
    per_horizon_huber = _huber_loss(err_n, delta=huber_delta)  # scalar over batch
    # per-horizon (keep for logging): Huber reduced over batch only, per component
    ph_huber = torch.stack([
        _huber_loss(err_n[:, i, :], delta=huber_delta) for i in range(H)
    ])  # (H,)
    position_loss = (ph_huber * w).sum() / w.sum()

    # --- 2) Physical reconstruction for aux terms + the reported metric ---
    pred_pos = reconstruct_positions(delta, cur_pos, horizons, target_mode)   # (B,H,2)
    true_pos = reconstruct_positions(target, cur_pos, horizons, target_mode)
    dist_km = haversine_km(pred_pos[..., 0], pred_pos[..., 1],
                           true_pos[..., 0], true_pos[..., 1])                # (B,H)
    per_horizon_km = dist_km.mean(dim=0)                                       # (H,)

    # --- 3) Optional physics / trajectory-consistency terms (kept near zero) ---
    prev_pos = torch.cat([cur_pos[:, None, :], pred_pos[:, :-1, :]], dim=1)
    seg_km = haversine_km(prev_pos[..., 0], prev_pos[..., 1],
                          pred_pos[..., 0], pred_pos[..., 1])                 # (B,H)
    prev_h = torch.cat([torch.zeros(1, device=horizons_t.device, dtype=horizons_t.dtype),
                        horizons_t[:-1]])
    seg_dur = (horizons_t - prev_h).clamp_min(1e-3).unsqueeze(0)
    seg_speed = seg_km / seg_dur                                              # (B,H) km/h

    speed_penalty = (torch.relu(seg_speed - max_plausible_speed_kmh) ** 2).mean()
    if H >= 2:
        dspeed = (seg_speed[:, 1:] - seg_speed[:, :-1]).abs()
        traj_penalty = (torch.relu(dspeed - max_segment_speed_change_kmh) ** 2).mean()
    else:
        traj_penalty = torch.zeros((), device=delta.device)

    accel_penalty = torch.zeros((), device=delta.device)
    if H >= 3 and acceleration_weight > 0:
        u_seg = 111.32 * pred_pos[..., 1].diff(dim=1, prepend=cur_pos[:, 1:2].expand(-1, 1)) \
            * torch.cos(torch.deg2rad(pred_pos[..., 0])).clamp_min(0.1)
        v_seg = 111.32 * pred_pos[..., 0].diff(dim=1, prepend=cur_pos[:, 0:1].expand(-1, 1))
        dur = horizons_t.diff(prepend=torch.zeros(1, dtype=horizons_t.dtype, device=horizons_t.device))
        u_seg_v = u_seg / dur.clamp_min(1e-3).unsqueeze(0)
        v_seg_v = v_seg / dur.clamp_min(1e-3).unsqueeze(0)
        dvec = torch.sqrt((u_seg_v[:, 1:] - u_seg_v[:, :-1]) ** 2
                          + (v_seg_v[:, 1:] - v_seg_v[:, :-1]) ** 2)
        accel_penalty = (torch.relu(dvec - max_segment_speed_change_kmh) ** 2).mean()

    turn_penalty = torch.zeros((), device=delta.device)
    if H >= 2 and turning_weight > 0:
        heading = _segment_angle_deg(pred_pos, cur_pos)                        # (B,H)
        dheading = (heading[:, 1:] - heading[:, :-1]).abs()
        dheading = torch.abs((dheading + 180.0) % 360.0 - 180.0)
        turn_penalty = (torch.relu(dheading - max_turn_deg) ** 2).mean()

    total = (position_loss
             + speed_consistency_weight * speed_penalty
             + trajectory_consistency_weight * traj_penalty
             + acceleration_weight * accel_penalty
             + turning_weight * turn_penalty)

    components = {
        "loss_total": total.item(),
        "loss_position": position_loss.item(),
        "loss_speed_consistency": speed_penalty.item(),
        "loss_trajectory_consistency": traj_penalty.item(),
        "loss_acceleration": accel_penalty.item(),
        "loss_turning": turn_penalty.item(),
        "mean_haversine_km": dist_km.mean().item(),
        "per_horizon_km": {float(h): float(per_horizon_km[i].item()) for i, h in enumerate(horizons)},
    }
    return total, components


# ---------------------------------------------------------------------------
# V7 loss: magnitude + direction (distance_direction) target.
#
# The V7 model predicts, per horizon:
#   delta[h] = ( distance_km, direction_sin, direction_cos )
# where distance is great-circle km from the CURRENT position and
# (sin, cos) encode the great-circle initial bearing (clockwise from north).
# Targets are CUMULATIVE (current -> +h), never incremental between horizons.
#
# Components (spec #7, #8, #13, #14):
#   * distance loss : Huber/SmoothL1 on the NORMALIZED distance error
#                     (divided by a per-horizon training-set distance scale so a
#                     24h horizon does not numerically dominate just because its
#                     displacement magnitude is larger).
#   * direction loss: Huber on direction_sin error + Huber on direction_cos
#                     error (no ordinary angular subtraction).
#   * long-horizon distance calibration (12h/24h), optional small weight 0.05:
#                     symmetric normalized distance error on those horizons only,
#                     trained (never a fixed post-hoc multiplier).
#   * velocity supervision, optional small weight 0.01: Huber on the NORMALIZED
#                     mean velocity (distance / horizon_hours) error.
#   * physics/turning/acceleration penalties are NOT used by default (weights 0).
# ---------------------------------------------------------------------------
V7_DEFAULT_HORIZON_WEIGHTS = [1.0, 1.0, 1.5, 2.0]   # [3h, 6h, 12h, 24h]
V7_LONG_HORIZONS = (12, 24)                          # hours used for calibration


def v7_loss(pred, target, cur_pos, horizons, target_mode="distance_direction",
            horizon_weights=None, horizon_scales=None, velocity_scales=None,
            huber_delta=1.0, long_horizon_distance_weight=0.05,
            velocity_loss_weight=0.01, eps=1e-6):
    """
    pred     : dict -> {"delta": (B,H,3)} = (pred_dist, pred_sin, pred_cos).
    target   : (B,H,3) = (true_dist, true_sin, true_cos) km / unit vector.
    cur_pos  : (B,2) anchor (lat, lon) degrees (for geographic reconstruction + metric).
    horizons : (H,) actual forecast lead hours.
    horizon_weights   : (H,) moderate per-horizon weights, default [1,1,1.5,2].
    horizon_scales    : (H,) per-horizon distance scale (km) from training set
                        (e.g. median true distance), used to normalize the
                        distance loss and the velocity loss. Default 1.0.
    velocity_scales   : (H,) per-horizon mean-velocity scale (km/h) from training
                        set, used to normalize the velocity loss. Default derived
                        from horizon_scales / horizon_hours.
    long_horizon_distance_weight : small weight (0.05) for the 12h/24h distance
                        calibration term.
    velocity_loss_weight : small weight (0.01) for the velocity supervision term.

    TOTAL = (weighted per-horizon [distance + direction] loss)
          + long_horizon_distance_weight * long-horizon distance calibration
          + velocity_loss_weight          * velocity supervision

    Returns (total, components) including the physical per-horizon Haversine km
    (the reported evaluation metric) and reconstruction diagnostics.
    """
    from dataset import reconstruct_positions
    delta = pred["delta"]                      # (B, H, 3)
    H = delta.shape[1]
    horizons_t = torch.as_tensor(horizons, dtype=delta.dtype, device=delta.device)
    if horizon_weights is None:
        horizon_weights = V7_DEFAULT_HORIZON_WEIGHTS[:H]
    w = torch.as_tensor(horizon_weights, dtype=delta.dtype, device=delta.device)
    assert w.shape[0] == H

    pred_dist = delta[..., 0].clamp_min(0.0)
    pred_sin = delta[..., 1]
    pred_cos = delta[..., 2]
    true_dist = target[..., 0]
    true_sin = target[..., 1]
    true_cos = target[..., 2]

    if horizon_scales is None:
        scale = torch.ones(H, dtype=delta.dtype, device=delta.device)
    else:
        scale = torch.as_tensor(horizon_scales, dtype=delta.dtype, device=delta.device).clamp_min(eps)
        assert scale.shape[0] == H

    # --- 1) normalized distance loss (Huber), per horizon ---
    dist_err = (pred_dist - true_dist) / scale.view(1, H)
    ph_dist = torch.stack([
        _huber_loss(dist_err[:, i], delta=huber_delta) for i in range(H)
    ])  # (H,)

    # --- 2) direction loss: Huber on sin + Huber on cos (per horizon) ---
    ph_dir = torch.stack([
        0.5 * (_huber_loss(pred_sin[:, i] - true_sin[:, i], delta=huber_delta)
               + _huber_loss(pred_cos[:, i] - true_cos[:, i], delta=huber_delta))
        for i in range(H)
    ])  # (H,)

    distance_direction_loss = ((ph_dist + ph_dir) * w).sum() / w.sum()

    # --- 3) long-horizon distance calibration (12h/24h), normalized ---
    calib_idx = [i for i, h in enumerate(horizons) if int(h) in V7_LONG_HORIZONS]
    if calib_idx and long_horizon_distance_weight > 0:
        idx = calib_idx
        c_err = (pred_dist[:, idx] - true_dist[:, idx]) / scale.view(1, H)[:, idx]
        long_horizon_loss = _huber_loss(c_err, delta=huber_delta)
    else:
        long_horizon_loss = torch.zeros((), device=delta.device)

    # --- 4) velocity supervision (distance / elapsed hours), normalized ---
    if velocity_loss_weight > 0:
        if velocity_scales is None:
            vscale = (scale / horizons_t.clamp_min(eps).to(scale.dtype)).clamp_min(eps)
        else:
            vscale = torch.as_tensor(velocity_scales, dtype=delta.dtype,
                                     device=delta.device).clamp_min(eps)
        pred_vel = pred_dist / horizons_t.to(delta.dtype).view(1, H).clamp_min(eps)
        true_vel = true_dist / horizons_t.to(delta.dtype).view(1, H).clamp_min(eps)
        vel_err = (pred_vel - true_vel) / vscale.view(1, H)
        velocity_loss = _huber_loss(vel_err, delta=huber_delta)
    else:
        velocity_loss = torch.zeros((), device=delta.device)

    total = (distance_direction_loss
             + long_horizon_distance_weight * long_horizon_loss
             + velocity_loss_weight * velocity_loss)

    # --- reported metric: physical per-horizon Haversine km ---
    pred_pos = reconstruct_positions(delta, cur_pos, horizons, target_mode)   # (B,H,2)
    true_pos = reconstruct_positions(target, cur_pos, horizons, target_mode)
    dist_km = haversine_km(pred_pos[..., 0], pred_pos[..., 1],
                           true_pos[..., 0], true_pos[..., 1])                # (B,H)
    per_horizon_km = dist_km.mean(dim=0)                                       # (H,)

    components = {
        "loss_total": total.item(),
        "loss_distance_direction": distance_direction_loss.item(),
        "loss_long_horizon": long_horizon_loss.item(),
        "loss_velocity": velocity_loss.item(),
        "mean_haversine_km": dist_km.mean().item(),
        "per_horizon_km": {float(h): float(per_horizon_km[i].item()) for i, h in enumerate(horizons)},
        "pred_distance_km": pred_dist.detach().mean(dim=0),
        "true_distance_km": true_dist.detach().mean(dim=0),
    }
    return total, components


# ---------------------------------------------------------------------------
# V9 loss: V5-preserving primary dx/dy + auxiliary magnitude / direction
# supervision + learned bounded calibration.
#
# V9 keeps V5's winning primary target (dx_km east, dy_km north) and its
# weighted per-horizon Haversine position loss. It then adds three controlled
# auxiliary terms designed from the V8 diagnostic (primary = long-horizon
# magnitude underprediction; secondary = direction error):
#
#   * magnitude_aux_weight * magnitude_loss
#       SmoothL1(normalized_pred_log_distance, normalized_true_log_distance),
#       where true_log_distance = log1p(sqrt(true_dx^2 + true_dy^2)) normalized
#       with TRAINING-SET-only statistics. Gives explicit magnitude supervision.
#   * direction_aux_weight * direction_loss
#       SmoothL1(pred_sin, true_sin) + SmoothL1(pred_cos, true_cos), where the
#       target direction is bearing = atan2(dx, dy) (east/north convention),
#       target_sin = sin(bearing), target_cos = cos(bearing). Only applied where
#       true_distance_km > direction_min_distance_km (bearing is unstable for
#       near-stationary fixes). Auxiliary only -- never dominant.
#   * calibration_regularization_weight * calibration_reg
#       mean((scale - 1)^2) kept small so the model does not over-rely on the
#       learned calibration head.
#
# The PRIMARY trajectory loss is computed on the CALIBRATED delta (raw dx/dy
# scaled by the learned per-horizon scale), because that is what feeds
# geographic reconstruction and the reported Haversine metric. The scale is a
# bounded, learnable per-horizon correction (1 + max_correction*tanh(raw)),
# trained from the data -- never a fixed post-hoc test-set multiplier.
# ---------------------------------------------------------------------------
V9_DEFAULT_HORIZON_WEIGHTS = [1.0, 1.0, 1.5, 2.0]  # [3h, 6h, 12h, 24h]


def v9_loss(pred, target, cur_pos, horizons, target_mode="km",
            horizon_weights=None, log_distance_scales=None,
            huber_delta=1.0, nll_weight=0.1,
            magnitude_aux_weight=0.05, direction_aux_weight=0.02,
            direction_min_distance_km=10.0,
            calibration_regularization_weight=0.01,
            max_plausible_speed_kmh=DEFAULT_MAX_PLAUSIBLE_SPEED_KMH,
            eps=1e-6):
    """Combined V9 loss.

    pred     : dict -> {"delta", "delta_calibrated", "log_var",
                        "magnitude_pred", "direction_pred", "scale"}
    target   : (B,H,2) true km target = (true_dx_km east, true_dy_km north).
    cur_pos  : (B,2) anchor (lat, lon) degrees.
    horizons : (H,) actual forecast lead hours.
    horizon_weights : (H,) moderate per-horizon weights, default [1,1,1.5,2].
    log_distance_scales : (H,) per-horizon std of log1p(distance) from the
                        training set, used to normalize the magnitude target
                        (None -> 1.0).
    nll_weight : small uncertainty weight (matches V5).
    magnitude_aux_weight : default 0.05.
    direction_aux_weight : default 0.02.
    direction_min_distance_km : threshold below which direction loss is ignored.

    TOTAL = weighted per-horizon Haversine loss on the CALIBRATED dx/dy
          + nll_weight   * Gaussian NLL in km space (on raw delta)
          + magnitude_aux_weight * normalized log-distance SmoothL1
          + direction_aux_weight * sin/cos SmoothL1 (masked near-stationary)
          + calibration_regularization_weight * mean((scale-1)^2)
          + speed_consistency penalty (mild, on calibrated trajectory)

    Returns (total, components) incl. raw vs calibrated per-horizon km.
    """
    from dataset import reconstruct_positions
    delta_raw = pred["delta"]                    # (B,H,2) raw km
    delta_cal = pred["delta_calibrated"]         # (B,H,2) calibrated km
    log_var = torch.clamp(pred["log_var"], min=-10.0, max=5.0)
    mag_pred = pred["magnitude_pred"]            # (B,H,1) normalized log-distance
    dir_pred = pred["direction_pred"]            # (B,H,2) (sin, cos)
    scale = pred["scale"]                        # (B,H,1)

    H = delta_raw.shape[1]
    horizons_t = torch.as_tensor(horizons, dtype=delta_raw.dtype, device=delta_raw.device)
    if horizon_weights is None:
        horizon_weights = V9_DEFAULT_HORIZON_WEIGHTS[:H]
    w = torch.as_tensor(horizon_weights, dtype=delta_raw.dtype, device=delta_raw.device)
    assert w.shape[0] == H

    # --- True magnitude + direction in km space ---
    true_dx = target[..., 0]
    true_dy = target[..., 1]
    true_dist = torch.sqrt(true_dx ** 2 + true_dy ** 2 + eps)        # (B,H) km
    true_bearing = torch.atan2(true_dx, true_dy)                     # rad (east/north)
    true_sin = torch.sin(true_bearing)
    true_cos = torch.cos(true_bearing)

    # --- 1) PRIMARY: weighted per-horizon Haversine loss on CALIBRATED dx/dy ---
    pred_pos = reconstruct_positions(delta_cal, cur_pos, horizons, "km")   # (B,H,2)
    true_pos = reconstruct_positions(target, cur_pos, horizons, "km")
    dist_km = haversine_km(pred_pos[..., 0], pred_pos[..., 1],
                           true_pos[..., 0], true_pos[..., 1])             # (B,H)
    per_horizon_km = dist_km.mean(dim=0)
    trajectory_loss = (per_horizon_km * w).sum() / w.sum()

    # Raw (pre-calibration) trajectory for the raw-vs-calibrated diagnostic.
    raw_pos = reconstruct_positions(delta_raw, cur_pos, horizons, "km")
    raw_dist_km = haversine_km(raw_pos[..., 0], raw_pos[..., 1],
                               true_pos[..., 0], true_pos[..., 1])         # (B,H)
    per_horizon_raw_km = raw_dist_km.mean(dim=0)

    # --- 2) Heteroscedastic NLL in the model's km space (on raw delta) ---
    sq_err = (delta_raw - target) ** 2
    var = torch.exp(log_var) + eps
    nll = (0.5 * (sq_err / var) + 0.5 * log_var).mean()

    # --- 3) Auxiliary magnitude loss: SmoothL1 on normalized log-distance ---
    if log_distance_scales is None:
        lscale = torch.ones(H, dtype=delta_raw.dtype, device=delta_raw.device)
    else:
        lscale = torch.as_tensor(log_distance_scales, dtype=delta_raw.dtype,
                                 device=delta_raw.device).clamp_min(eps)
    true_log_dist = torch.log1p(true_dist)                                  # (B,H)
    mag_target = (true_log_dist - 0.0) / lscale.view(1, H)                  # (B,H)
    mag_err = mag_pred[..., 0] - mag_target
    magnitude_loss = torch.nn.functional.huber_loss(
        mag_err, torch.zeros_like(mag_err), delta=huber_delta)

    # --- 4) Auxiliary direction loss: SmoothL1(sin)+SmoothL1(cos), masked ---
    moving = (true_dist > direction_min_distance_km).float()                # (B,H)
    if direction_aux_weight > 0 and moving.sum() > 0:
        sin_err = (dir_pred[..., 0] - true_sin) * moving
        cos_err = (dir_pred[..., 1] - true_cos) * moving
        per_el = torch.nn.functional.huber_loss(sin_err, torch.zeros_like(sin_err),
                                                delta=huber_delta, reduction="none")
        per_el = per_el + torch.nn.functional.huber_loss(
            cos_err, torch.zeros_like(cos_err), delta=huber_delta, reduction="none")
        direction_loss = (0.5 * per_el * moving).sum() / moving.sum().clamp_min(1.0)
    else:
        direction_loss = torch.zeros((), device=delta_raw.device)

    # --- 5) Calibration regularization: keep scale close to 1 ---
    calibration_reg = ((scale - 1.0) ** 2).mean()

    # --- 6) Mild speed-plausibility penalty on the calibrated trajectory ---
    prev_pos = torch.cat([cur_pos[:, None, :], pred_pos[:, :-1, :]], dim=1)
    seg_km = haversine_km(prev_pos[..., 0], prev_pos[..., 1],
                          pred_pos[..., 0], pred_pos[..., 1])               # (B,H)
    prev_h = torch.cat([torch.zeros(1, device=horizons_t.device,
                                    dtype=horizons_t.dtype), horizons_t[:-1]])
    seg_dur = (horizons_t - prev_h).clamp_min(1e-3).unsqueeze(0)
    seg_speed = seg_km / seg_dur                                             # km/h
    speed_penalty = (torch.relu(seg_speed - max_plausible_speed_kmh) ** 2).mean()

    total = (
        trajectory_loss
        + nll_weight * nll
        + magnitude_aux_weight * magnitude_loss
        + direction_aux_weight * direction_loss
        + calibration_regularization_weight * calibration_reg
        + 0.01 * speed_penalty
    )

    components = {
        "loss_total": total.item(),
        "loss_trajectory": trajectory_loss.item(),
        "loss_nll": nll.item(),
        "loss_magnitude": magnitude_loss.item(),
        "loss_direction": direction_loss.item(),
        "loss_calibration_reg": calibration_reg.item(),
        "loss_speed": speed_penalty.item(),
        "mean_haversine_km": dist_km.mean().item(),
        "mean_raw_haversine_km": raw_dist_km.mean().item(),
        "per_horizon_km": {float(h): float(per_horizon_km[i].item()) for i, h in enumerate(horizons)},
        "per_horizon_raw_km": {float(h): float(per_horizon_raw_km[i].item()) for i, h in enumerate(horizons)},
        "scale_mean": float(scale.mean().item()),
    }
    return total, components


# ---------------------------------------------------------------------------
# V11 loss: identifiable magnitude calibration.
#
# V10 failed because the raw dx/dy head and the learned scale head could trade
# off (raw collapsed to ~0.48x, scale saturated at 2.0). V11 restores
# identifiability with three terms:
#
#   * primary  : weighted per-horizon Haversine loss on the CALIBRATED dx/dy
#                (the final prediction), exactly matching V5's evaluation metric.
#   * anchor   : RAW-magnitude anchoring. SmoothL1 on the NORMALIZED raw
#                distance error,
#                    norm_raw_dist = sqrt(dx_raw^2 + dy_raw^2) / raw_dist_scale
#                    norm_true_dist= sqrt(dx_true^2 + dy_true^2) / raw_dist_scale
#                with per-horizon scales computed on the TRAINING SET ONLY
#                (median true distance). EXPRESSED ON THE RAW OUTPUT so the
#                raw head is explicitly pinned near the true displacement and
#                cannot shrink while the scale compensates.
#   * scale_reg : 0.01 * mean((scale - 1)^2) keeps the correction modest.
#
# The optional V9 magnitude_auxiliary (normalized log-distance) and direction
# aux heads are supported but BOTH default to 0.0 for the first V11 experiment,
# so the anchor effect is isolated (controlled experiment per spec #9, #10).
#
# Loss concept (spec #20):
#   total = weighted_calibrated_position_loss
#         + raw_magnitude_anchor_weight * raw_magnitude_anchor_loss
#         + scale_reg_weight * scale_regularization
#         (+ magnitude_aux_weight * magnitude_aux_loss)   # 0 by default
#         (+ direction_aux_weight * direction_aux_loss)   # 0 by default
# ---------------------------------------------------------------------------
V11_DEFAULT_HORIZON_WEIGHTS = [1.0, 1.0, 1.5, 2.0]  # [3h, 6h, 12h, 24h]


def v11_loss(pred, target, cur_pos, horizons, target_mode="km",
             horizon_weights=None, raw_distance_scales=None,
             log_distance_scales=None, huber_delta=1.0, nll_weight=0.0,
             raw_magnitude_anchor_weight=0.05,
             scale_reg_weight=0.01,
             magnitude_aux_weight=0.0, direction_aux_weight=0.0,
             direction_min_distance_km=10.0, eps=1e-6):
    """Combined V11 loss.

    pred     : dict -> {"delta", "delta_calibrated", "log_var",
                        "magnitude_pred", "direction_pred", "scale"}
    target   : (B,H,2) true km target = (dx_km east, dy_km north).
    cur_pos  : (B,2) anchor (lat, lon) degrees.
    horizons : (H,) actual forecast lead hours.
    horizon_weights : (H,) per-horizon weights for the calibrated position loss,
                      default [1,1,1.5,2] (spec #12).
    raw_distance_scales : (H,) per-horizon scale (km) of TRUE displacement from
                      the TRAINING SET used to normalize the raw-magnitude anchor.
                      None -> uses a batch-computed fallback (norm by batch mean;
                      NOT recommended for training -- always pass training stats).
    log_distance_scales : (H,) per-horizon std of log1p(distance) from training
                      set for the (optional, default-off) magnitude aux head.
    raw_magnitude_anchor_weight : default 0.05 (spec #6).
    scale_reg_weight : default 0.01 (spec #13).
    magnitude_aux_weight : default 0.0 (disabled; spec #9).
    direction_aux_weight : default 0.0 (disabled; spec #10).

    Returns (total, components) with raw vs calibrated per-horizon metrics.
    """
    from dataset import reconstruct_positions
    delta_raw = pred["delta"]                    # (B,H,2) raw km
    delta_cal = pred["delta_calibrated"]         # (B,H,2) calibrated km
    log_var = torch.clamp(pred["log_var"], min=-10.0, max=5.0)
    mag_pred = pred["magnitude_pred"]            # (B,H,1) normalized log-distance
    dir_pred = pred["direction_pred"]            # (B,H,2) (sin, cos)
    scale = pred["scale"]                        # (B,H,1)

    H = delta_raw.shape[1]
    horizons_t = torch.as_tensor(horizons, dtype=delta_raw.dtype, device=delta_raw.device)
    if horizon_weights is None:
        horizon_weights = V11_DEFAULT_HORIZON_WEIGHTS[:H]
    w = torch.as_tensor(horizon_weights, dtype=delta_raw.dtype, device=delta_raw.device)
    assert w.shape[0] == H

    true_dx = target[..., 0]
    true_dy = target[..., 1]
    true_dist = torch.sqrt(true_dx ** 2 + true_dy ** 2 + eps)      # (B,H) km
    raw_dx = delta_raw[..., 0]
    raw_dy = delta_raw[..., 1]
    raw_dist = torch.sqrt(raw_dx ** 2 + raw_dy ** 2 + eps)         # (B,H) km

    # --- 1) PRIMARY: weighted per-horizon Haversine loss on CALIBRATED dx/dy ---
    pred_pos = reconstruct_positions(delta_cal, cur_pos, horizons, "km")   # (B,H,2)
    true_pos = reconstruct_positions(target, cur_pos, horizons, "km")
    dist_km = haversine_km(pred_pos[..., 0], pred_pos[..., 1],
                           true_pos[..., 0], true_pos[..., 1])             # (B,H)
    per_horizon_km = dist_km.mean(dim=0)
    trajectory_loss = (per_horizon_km * w).sum() / w.sum()

    # Raw (pre-calibration) trajectory for the raw-vs-calibrated diagnostic.
    raw_pos = reconstruct_positions(delta_raw, cur_pos, horizons, "km")
    raw_dist_km = haversine_km(raw_pos[..., 0], raw_pos[..., 1],
                               true_pos[..., 0], true_pos[..., 1])         # (B,H)
    per_horizon_raw_km = raw_dist_km.mean(dim=0)

    # --- 2) RAW-MAGNITUDE ANCHOR (spec #6, #7) ---
    # Normalize with TRAINING-SET-only per-horizon displacement scales so no
    # horizon dominates numerically and far displacements aren't overweighted.
    if raw_distance_scales is None:
        # Fallback only (avoids a hard crash in a smoke/plumbing test): normalize
        # by the batch mean of the true distance. Train-time must pass the real
        # training-set scales (the training harness always does).
        batch_scale = torch.mean(true_dist, dim=0) + eps          # (H,)
    else:
        batch_scale = torch.as_tensor(raw_distance_scales,
                                      dtype=delta_raw.dtype, device=delta_raw.device).clamp_min(eps)
        assert batch_scale.shape[0] == H
    err_anchor = (raw_dist - true_dist) / batch_scale.view(1, H)          # (B,H)
    raw_magnitude_anchor_loss = torch.nn.functional.huber_loss(
        err_anchor, torch.zeros_like(err_anchor), delta=huber_delta)

    # --- 3) SCALE REGULARIZATION (spec #13): keep scale centered near 1 ---
    scale_reg = ((scale - 1.0) ** 2).mean()

    # --- 4) Optional NLL on raw delta (kept cheap; default weight off) ---
    sq_err = (delta_raw - target) ** 2
    var = torch.exp(log_var) + eps
    nll = (0.5 * (sq_err / var) + 0.5 * log_var).mean()

    # --- 5) Optional magnitude aux head (log-distance; default off) ---
    magnitude_loss = torch.zeros((), device=delta_raw.device)
    if magnitude_aux_weight > 0:
        if log_distance_scales is None:
            lscale = torch.ones(H, dtype=delta_raw.dtype, device=delta_raw.device)
        else:
            lscale = torch.as_tensor(log_distance_scales,
                                     dtype=delta_raw.dtype, device=delta_raw.device).clamp_min(eps)
        mag_target = torch.log1p(true_dist) / lscale.view(1, H)
        mag_err = mag_pred[..., 0] - mag_target
        magnitude_loss = torch.nn.functional.huber_loss(
            mag_err, torch.zeros_like(mag_err), delta=huber_delta)

    # --- 6) Optional direction aux head (sin/cos; default off) ---
    direction_loss = torch.zeros((), device=delta_raw.device)
    if direction_aux_weight > 0:
        true_bearing = torch.atan2(true_dx, true_dy)                 # rad (east/north)
        true_sin = torch.sin(true_bearing)
        true_cos = torch.cos(true_bearing)
        moving = (true_dist > direction_min_distance_km).float()
        if moving.sum() > 0:
            sin_err = (dir_pred[..., 0] - true_sin) * moving
            cos_err = (dir_pred[..., 1] - true_cos) * moving
            per_el = torch.nn.functional.huber_loss(
                sin_err, torch.zeros_like(sin_err), delta=huber_delta, reduction="none")
            per_el = per_el + torch.nn.functional.huber_loss(
                cos_err, torch.zeros_like(cos_err), delta=huber_delta, reduction="none")
            direction_loss = (0.5 * per_el * moving).sum() / moving.sum().clamp_min(1.0)

    total = (
        trajectory_loss
        + nll_weight * nll
        + raw_magnitude_anchor_weight * raw_magnitude_anchor_loss
        + scale_reg_weight * scale_reg
        + magnitude_aux_weight * magnitude_loss
        + direction_aux_weight * direction_loss
    )

    components = {
        "loss_total": total.item(),
        "loss_trajectory": trajectory_loss.item(),
        "loss_raw_anchor": raw_magnitude_anchor_loss.item(),
        "loss_scale_reg": scale_reg.item(),
        "loss_nll": nll.item(),
        "loss_magnitude": magnitude_loss.item(),
        "loss_direction": direction_loss.item(),
        "mean_haversine_km": dist_km.mean().item(),
        "mean_raw_haversine_km": raw_dist_km.mean().item(),
        "per_horizon_km": {float(h): float(per_horizon_km[i].item()) for i, h in enumerate(horizons)},
        "per_horizon_raw_km": {float(h): float(per_horizon_raw_km[i].item()) for i, h in enumerate(horizons)},
        "scale_mean": float(scale.mean().item()),
    }
    return total, components


# ---------------------------------------------------------------------------
# V12 loss: V11 identifiable calibration + fine 2h-horizon accuracy terms.
#
# V12 reuses V11's model (CycloneTransformerV11, per-horizon bounded scale in
# [0.80, 1.20] zero-initialised to exactly 1.0) and its identifiable
# raw-magnitude-anchor + scale-reg structure. On top of that it adds the
# accuracy levers for the finer 2h grid and monotonic physics:
#
#   * trajectory_consistency_weight * trajectory-consistency penalty
#       Per-segment speed-CHANGE exceedance across EVERY adjacent horizon pair
#       (H-1 pairs; for V12's 12 horizons that is all 11 pairs). Soft hinge,
#       identical in style to the speed/turn hinges already in V3/V5.
#   * monotonic_distance_weight * monotonic-distance hinge
#       Cumulative calibrated displacement magnitude (km from the current fix)
#       must not DECREASE as the horizon grows: relu(disp[h-1] - disp[h])^2
#       averaged over adjacent pairs (a storm cannot come back toward the
#       starting point; it may only slow, not reverse the running displacement).
#   * monotonic_uncertainty_weight * monotonic-uncertainty hinge
#       The predicted std  sigma = sqrt(exp(log_var))  must be non-decreasing
#       in horizon: relu(sigma[h-1] - sigma[h])^2. Longer lead times must be
#       (weakly) more uncertain than shorter ones.
#
# The PRIMARY trajectory loss stays the weighted per-horizon Haversine on the
# CALIBRATED dx/dy (matches the reported metric). All the V11 terms (NLL,
# raw-magnitude anchor, scale reg, aux heads) carry through; the V12 additions
# are soft regularizers that never dominate the position loss.
# ---------------------------------------------------------------------------
V12_DEFAULT_HORIZON_WEIGHTS = [1.0, 1.0, 1.5, 2.0]  # placeholder (formula default lives in v12_train)


def v12_loss(pred, target, cur_pos, horizons, target_mode="km",
             horizon_weights=None, raw_distance_scales=None,
             log_distance_scales=None, huber_delta=1.0, nll_weight=0.0,
             raw_magnitude_anchor_weight=0.05, scale_reg_weight=0.01,
             magnitude_aux_weight=0.0, direction_aux_weight=0.0,
             direction_min_distance_km=10.0,
             trajectory_consistency_weight=0.05,
             monotonic_distance_weight=0.02,
             monotonic_uncertainty_weight=0.02,
             max_segment_speed_change_kmh=DEFAULT_MAX_SEGMENT_SPEED_CHANGE_KMH,
             eps=1e-6):
    """Combined V12 loss (V11 identifiable calibration + monotonic/consistency).

    pred     : dict -> {"delta", "delta_calibrated", "log_var",
                        "magnitude_pred", "direction_pred", "scale"}
    target   : (B,H,2) true km target = (dx_km east, dy_km north).
    cur_pos  : (B,2) anchor (lat, lon) degrees.
    horizons : (H,) actual forecast lead hours (V12: 12 horizons, 2h apart).
    horizon_weights : (H,) per-horizon weights for the calibrated position loss
                      (V12 default is the formula-driven weights from v12_train).
    raw_distance_scales / log_distance_scales : training-set-only scales for the
                      raw-magnitude anchor / optional aux magnitude head.
    trajectory_consistency_weight : soft hinge on extreme per-segment speed-change
                      over all adjacent horizon pairs (H-1 pairs).
    monotonic_distance_weight   : soft hinge that cumulative displacement must not
                      decrease with horizon.
    monotonic_uncertainty_weight: soft hinge that predicted sigma must not decrease
                      with horizon.

    Returns (total, components) with raw vs calibrated per-horizon metrics and
    the three V12 physical/monotonic components.
    """
    from dataset import reconstruct_positions
    delta_raw = pred["delta"]                    # (B,H,2) raw km
    delta_cal = pred["delta_calibrated"]         # (B,H,2) calibrated km
    lv = torch.clamp(pred["log_var"], min=-10.0, max=5.0)
    mag_pred = pred["magnitude_pred"]            # (B,H,1)
    dir_pred = pred["direction_pred"]            # (B,H,2)
    scale = pred["scale"]                        # (B,H,1)

    H = delta_raw.shape[1]
    horizons_t = torch.as_tensor(horizons, dtype=delta_raw.dtype, device=delta_raw.device)
    if horizon_weights is None:
        horizon_weights = [1.0] * H
    w = torch.as_tensor(horizon_weights, dtype=delta_raw.dtype, device=delta_raw.device)
    assert w.shape[0] == H

    true_dx = target[..., 0]
    true_dy = target[..., 1]
    true_dist = torch.sqrt(true_dx ** 2 + true_dy ** 2 + eps)      # (B,H) km
    raw_dx = delta_raw[..., 0]
    raw_dy = delta_raw[..., 1]
    raw_dist = torch.sqrt(raw_dx ** 2 + raw_dy ** 2 + eps)         # (B,H) km

    # --- 1) PRIMARY: weighted per-horizon Haversine on CALIBRATED dx/dy ---
    pred_pos = reconstruct_positions(delta_cal, cur_pos, horizons, "km")
    true_pos = reconstruct_positions(target, cur_pos, horizons, "km")
    dist_km = haversine_km(pred_pos[..., 0], pred_pos[..., 1],
                           true_pos[..., 0], true_pos[..., 1])      # (B,H)
    per_horizon_km = dist_km.mean(dim=0)
    trajectory_loss = (per_horizon_km * w).sum() / w.sum()

    raw_pos = reconstruct_positions(delta_raw, cur_pos, horizons, "km")
    raw_dist_km = haversine_km(raw_pos[..., 0], raw_pos[..., 1],
                               true_pos[..., 0], true_pos[..., 1])  # (B,H)
    per_horizon_raw_km = raw_dist_km.mean(dim=0)

    # --- 2) RAW-MAGNITUDE ANCHOR (training-set scales) ---
    if raw_distance_scales is None:
        batch_scale = torch.mean(true_dist, dim=0) + eps
    else:
        batch_scale = torch.as_tensor(raw_distance_scales, dtype=delta_raw.dtype,
                                      device=delta_raw.device).clamp_min(eps)
    err_anchor = (raw_dist - true_dist) / batch_scale.view(1, H)
    raw_magnitude_anchor_loss = torch.nn.functional.huber_loss(
        err_anchor, torch.zeros_like(err_anchor), delta=huber_delta)

    # --- 3) SCALE REGULARIZATION ---
    scale_reg = ((scale - 1.0) ** 2).mean()

    # --- 4) Heteroscedastic NLL in km space (kept cheap; default small) ---
    sq_err = (delta_raw - target) ** 2
    var = torch.exp(lv) + eps
    nll = (0.5 * (sq_err / var) + 0.5 * lv).mean()

    # --- 5) Optional aux magnitude/direction heads (default off) ---
    magnitude_loss = torch.zeros((), device=delta_raw.device)
    if magnitude_aux_weight > 0:
        if log_distance_scales is None:
            lscale = torch.ones(H, dtype=delta_raw.dtype, device=delta_raw.device)
        else:
            lscale = torch.as_tensor(log_distance_scales, dtype=delta_raw.dtype,
                                     device=delta_raw.device).clamp_min(eps)
        mag_target = torch.log1p(true_dist) / lscale.view(1, H)
        magnitude_loss = torch.nn.functional.huber_loss(
            mag_pred[..., 0] - mag_target, torch.zeros_like(mag_target), delta=huber_delta)

    direction_loss = torch.zeros((), device=delta_raw.device)
    if direction_aux_weight > 0:
        true_bearing = torch.atan2(true_dx, true_dy)
        true_sin = torch.sin(true_bearing)
        true_cos = torch.cos(true_bearing)
        moving = (true_dist > direction_min_distance_km).float()
        if moving.sum() > 0:
            sin_err = (dir_pred[..., 0] - true_sin) * moving
            cos_err = (dir_pred[..., 1] - true_cos) * moving
            per_el = torch.nn.functional.huber_loss(
                sin_err, torch.zeros_like(sin_err), delta=huber_delta, reduction="none")
            per_el = per_el + torch.nn.functional.huber_loss(
                cos_err, torch.zeros_like(cos_err), delta=huber_delta, reduction="none")
            direction_loss = (0.5 * per_el * moving).sum() / moving.sum().clamp_min(1.0)

    # --- 6) ADJACENT-HORIZON TRAJECTORY CONSISTENCY (all H-1 pairs) ---
    # Per-segment predicted speed over the CALIBRATED trajectory, then penalize
    # only EXTREME speed changes between adjacent horizons (V3/V5-style hinge).
    if H >= 2:
        prev_pos = torch.cat([cur_pos[:, None, :], pred_pos[:, :-1, :]], dim=1)
        seg_km = haversine_km(prev_pos[..., 0], prev_pos[..., 1],
                              pred_pos[..., 0], pred_pos[..., 1])        # (B,H)
        prev_h = torch.cat([torch.zeros(1, device=horizons_t.device,
                                        dtype=horizons_t.dtype), horizons_t[:-1]])
        seg_dur = (horizons_t - prev_h).clamp_min(1e-3).unsqueeze(0)
        seg_speed = seg_km / seg_dur                                      # (B,H) km/h
        dspeed = (seg_speed[:, 1:] - seg_speed[:, :-1]).abs()
        trajectory_consistency = (torch.relu(dspeed - max_segment_speed_change_kmh) ** 2).mean()
    else:
        trajectory_consistency = torch.zeros((), device=delta_raw.device)

    # --- 7) MONOTONIC DISPLACEMENT MAGNITUDE HINGE (cumulative, non-decreasing) ---
    # cal displacement magnitude from the CURRENT fix; a cyclone cannot come back
    # toward the start, so disp must not shrink as the horizon grows.
    cal_dx = delta_cal[..., 0]
    cal_dy = delta_cal[..., 1]
    cal_disp = torch.sqrt(cal_dx ** 2 + cal_dy ** 2 + eps)               # (B,H) km
    if H >= 2:
        drop_mag = torch.relu(cal_disp[:, :-1] - cal_disp[:, 1:])        # (B,H-1)
        monotonic_distance = (drop_mag ** 2).mean()
    else:
        monotonic_distance = torch.zeros((), device=delta_raw.device)

    # --- 8) MONOTONIC UNCERTAINTY HINGE (sigma non-decreasing with horizon) ---
    if H >= 2:
        sigma = torch.sqrt(torch.exp(lv) + eps).mean(dim=-1)             # (B,H)
        drop_unc = torch.relu(sigma[:, :-1] - sigma[:, 1:])              # (B,H-1)
        monotonic_uncertainty = (drop_unc ** 2).mean()
    else:
        monotonic_uncertainty = torch.zeros((), device=delta_raw.device)

    total = (
        trajectory_loss
        + nll_weight * nll
        + raw_magnitude_anchor_weight * raw_magnitude_anchor_loss
        + scale_reg_weight * scale_reg
        + magnitude_aux_weight * magnitude_loss
        + direction_aux_weight * direction_loss
        + trajectory_consistency_weight * trajectory_consistency
        + monotonic_distance_weight * monotonic_distance
        + monotonic_uncertainty_weight * monotonic_uncertainty
    )

    components = {
        "loss_total": total.item(),
        "loss_trajectory": trajectory_loss.item(),
        "loss_nll": nll.item(),
        "loss_raw_anchor": raw_magnitude_anchor_loss.item(),
        "loss_scale_reg": scale_reg.item(),
        "loss_magnitude": magnitude_loss.item(),
        "loss_direction": direction_loss.item(),
        "loss_trajectory_consistency": trajectory_consistency.item(),
        "loss_monotonic_distance": monotonic_distance.item(),
        "loss_monotonic_uncertainty": monotonic_uncertainty.item(),
        "mean_haversine_km": dist_km.mean().item(),
        "mean_raw_haversine_km": raw_dist_km.mean().item(),
        "per_horizon_km": {float(h): float(per_horizon_km[i].item()) for i, h in enumerate(horizons)},
        "per_horizon_raw_km": {float(h): float(per_horizon_raw_km[i].item()) for i, h in enumerate(horizons)},
        "scale_mean": float(scale.mean().item()),
    }
    return total, components


if __name__ == "__main__":
    B, H = 4, 4
    horizons = [3, 6, 12, 24]
    pred = {"delta": torch.randn(B, H, 2) * 0.1, "log_var": torch.zeros(B, H, 2)}
    target_delta = torch.randn(B, H, 2) * 0.1
    current_pos = torch.stack([torch.full((B,), 15.0), torch.full((B,), -50.0)], dim=1)
    loss, comp = physics_aware_loss(pred, target_delta, current_pos, horizon_hours=horizons,
                                     horizon_weights=[1.0, 1.0, 1.1, 1.2])
    print(loss.item(), comp)