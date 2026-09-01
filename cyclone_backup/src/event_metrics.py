"""Event-level performance metrics.

Row-level recall answers: "How many RI timestamps did we detect?"
But operationally you also care: "How many RI episodes/storms did we warn
about *before* RI?"  This module computes event-level metrics:

- RI-event detection rate (how many RI episodes were preceded by a warning)
- False alarms per storm
- Median warning lead time
- Missed RI events (storms with RI that were never warned)

An "RI episode" is defined as a contiguous block of RI-flagged timestamps
within a single storm.  A "warning" is any time the model's probability
exceeds the threshold *before* the episode starts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _contiguous_episodes(flags: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start_idx, end_idx), ...] of contiguous True runs."""
    episodes = []
    in_ep = False
    start = 0
    for i, f in enumerate(flags):
        if f and not in_ep:
            in_ep = True
            start = i
        elif not f and in_ep:
            in_ep = False
            episodes.append((start, i - 1))
    if in_ep:
        episodes.append((start, len(flags) - 1))
    return episodes


def event_level_metrics(
    df: pd.DataFrame,
    threshold: float,
    storm_col: str = "storm_id",
    time_col: str = "datetime_utc",
    target_col: str = "RI_24h",
    prob_col: str = "P_RI",
    warning_before_lead_h: float = 6.0,
) -> dict:
    """Compute event-level RI detection metrics.

    Args:
        df: DataFrame with storm_id, datetime_utc, RI_24h, P_RI.
        threshold: Decision threshold for "warning".
        storm_col: Storm ID column.
        time_col: Timestamp column.
        target_col: Binary RI label column.
        prob_col: Predicted probability column.
        warning_before_lead_h: Minimum lead time (hours) for a warning to
            count.  A warning at t qualifies for an RI episode starting at
            t + warning_before_lead_h or later.

    Returns:
        Dict with event-level metrics and per-storm details.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([storm_col, time_col]).reset_index(drop=True)
    df["warn"] = (df[prob_col] >= threshold).astype(int)

    total_ri_events = 0
    detected_ri_events = 0
    missed_storms = []
    total_false_alarms = 0
    lead_times = []
    per_storm = []

    for storm_id, grp in df.groupby(storm_col):
        grp = grp.sort_values(time_col).reset_index(drop=True)
        y = grp[target_col].to_numpy().astype(int)
        w = grp["warn"].to_numpy()
        times = grp[time_col].to_numpy()

        # Find RI episodes (contiguous blocks of RI=1).
        episodes = _contiguous_episodes(y == 1)
        n_ri = len(episodes)

        # Count false alarms: warn=1 but RI=0, within this storm.
        false_alarms = int(((w == 1) & (y == 0)).sum())
        total_false_alarms += false_alarms

        storm_detected = 0
        storm_missed = 0
        storm_leads = []

        for ep_start, ep_end in episodes:
            total_ri_events += 1
            ep_start_time = times[ep_start]

            # Was there a warning BEFORE the episode start (with lead time)?
            pre_mask = times[:ep_start] < ep_start_time
            pre_warn = w[:ep_start][pre_mask] if pre_mask.any() else np.array([])

            if pre_warn.any():
                # Find the last warning before the episode.
                warn_indices = np.where(pre_warn == 1)[0]
                if len(warn_indices) > 0:
                    last_warn_idx = warn_indices[-1]
                    warn_time = times[last_warn_idx]
                    lead_h = (ep_start_time - warn_time) / np.timedelta64(1, "h")
                    if lead_h >= warning_before_lead_h:
                        storm_detected += 1
                        storm_leads.append(float(lead_h))
                    else:
                        storm_missed += 1
                else:
                    storm_missed += 1
            else:
                storm_missed += 1

        detected_ri_events += storm_detected
        if n_ri > 0 and storm_detected == 0:
            missed_storms.append(str(storm_id))
        lead_times.extend(storm_leads)

        per_storm.append({
            "storm_id": str(storm_id),
            "n_ri_episodes": n_ri,
            "detected": storm_detected,
            "missed": storm_missed,
            "false_alarms": false_alarms,
            "median_lead_h": float(np.median(storm_leads)) if storm_leads else np.nan,
        })

    detection_rate = (detected_ri_events / total_ri_events
                      if total_ri_events > 0 else float("nan"))
    false_alarm_rate = (total_false_alarms / df[storm_col].nunique()
                        if df[storm_col].nunique() > 0 else float("nan"))

    return {
        "total_ri_episodes": total_ri_events,
        "detected_ri_episodes": detected_ri_events,
        "ri_event_detection_rate": float(detection_rate),
        "missed_ri_storms": missed_storms,
        "n_missed_storms": len(missed_storms),
        "total_false_alarms": total_false_alarms,
        "false_alarms_per_storm": float(false_alarm_rate),
        "median_warning_lead_time_h": float(np.median(lead_times)) if lead_times else float("nan"),
        "warning_lead_times_h": lead_times,
        "per_storm": per_storm,
    }
