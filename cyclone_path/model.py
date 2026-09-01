"""
model.py
--------
Architecture: numerical-feature MLP -> 2-layer Transformer encoder ->
multi-horizon (delta-lat, delta-lon, uncertainty) decoder.

Kept intentionally flat: one model class, no factories/registries/wrappers.
"""
import math
import torch
import torch.nn as nn

# Input feature order (must match dataset.FEATURE_COLS)
NUM_FEATURES = 9  # lat, lon, wind, mslp, rmw, sst, shear, speed_kmh, dt_hours
# note: bearing_sin/cos and month_sin/cos add 4 more -> 13 total, set below
FEATURE_DIM = 13


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for the input sequence."""

    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class CycloneTransformer(nn.Module):
    """
    Encodes an input window of storm timesteps and predicts, for each of
    `horizons`, the (delta_lat, delta_lon) displacement and its predicted
    log-variance (aleatoric uncertainty), relative to the last observed fix.

    Input : (batch, seq_len, feature_dim)
    Output: dict with
        "delta"   -> (batch, num_horizons, 2)   predicted (dlat, dlon) in degrees
        "log_var" -> (batch, num_horizons, 2)   predicted log-variance per component
    """

    def __init__(self, feature_dim=FEATURE_DIM, d_model=128, nhead=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1,
                 horizons=(1, 2, 4, 8)):
        super().__init__()
        self.horizons = list(horizons)
        num_horizons = len(self.horizons)

        # 1) Numerical feature MLP (per-timestep embedding)
        self.input_mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_enc = PositionalEncoding(d_model)

        # 2) 2-layer Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 3) Multi-horizon decoder head: one shared trunk, outputs per horizon
        #    4 numbers each -> (dlat, dlon, log_var_lat, log_var_lon)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_horizons * 4),
        )

    def forward(self, x, src_key_padding_mask=None):
        # x: (batch, seq_len, feature_dim)
        h = self.input_mlp(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)

        # Use the representation at the last real (non-padded) timestep of
        # each sequence. If no mask given, assume last position is valid.
        if src_key_padding_mask is not None:
            lengths = (~src_key_padding_mask).sum(dim=1) - 1  # index of last valid step
            summary = h[torch.arange(h.size(0)), lengths]
        else:
            summary = h[:, -1, :]

        out = self.decoder(summary)  # (batch, num_horizons*4)
        out = out.view(-1, len(self.horizons), 4)
        delta = out[..., 0:2]
        log_var = out[..., 2:4].clamp(min=-10.0, max=5.0)  # stability (matches losses.py clamp range)
        return {"delta": delta, "log_var": log_var}


# ---------------------------------------------------------------------------
# V5 model. The encoder is shared with the V3 architecture; only the decoder
# differs. The decoder is made target-representation agnostic: it outputs a
# (B, H, 2) value per horizon (interpreted in the current `target_mode` space
# by the loss) plus (B, H, 2) log-variance. Two decoder variants are provided:
#
#   "direct"     : MLP head, all horizons predicted independently (V3 default).
#   "sequential" : recurrent GRU cell that chains horizons Current -> +3h -> +6h
#                  -> +12h -> +24h. It emits per-segment increments in local km
#                  and accumulates them, so horizon predictions are consistent
#                  (no unrealistic jumps) and autoregressive. Supports teacher
#                  forcing at train time and free-run at inference.
# ---------------------------------------------------------------------------


class _SeqDecoder(nn.Module):
    """Recurrent decoder over forecast segments. Outputs, per segment, a local
    displacement increment (dx_km, dy_km); the cumulative sum is converted into
    the target space by the wrapper. With teacher forcing the accumulated state
    is reset to the true cumulative displacement before each step."""

    def __init__(self, d_model, horizons, dim_feedforward=256, dropout=0.0):
        super().__init__()
        self.horizons = list(horizons)
        self.pos_mlp = nn.Linear(2, d_model)
        self.dt_mlp = nn.Linear(1, d_model)
        # cell: combines hidden, position, segment-duration
        self.cell = nn.GRUCell(d_model + d_model + d_model, d_model)
        self.out_inc = nn.Linear(d_model, 2)      # (dx_km, dy_km) increment
        self.out_logvar = nn.Linear(d_model, 2)
        self.drop = nn.Dropout(dropout)

    def _horizons_t(self, device, dtype):
        return torch.as_tensor(self.horizons, dtype=dtype, device=device)

    def forward(self, h, cur_pos_deg, true_cum_km=None, tf_ratio=0.0,
                teacher_forcing=False):
        """h: (B, D) encoder summary. cur_pos_deg: (B, 2) anchor (lat, lon).
        true_cum_km: (B, H, 2) true cumulative (dx_km, dy_km) if available.
        Returns outputs (B,H,2) cumulative km, incs (B,H,2), log_var (B,H,2)."""
        B = h.shape[0]
        device, dtype = h.device, h.dtype
        H = len(self.horizons)
        horizons_t = self._horizons_t(device, dtype)
        prev_h = torch.cat([torch.zeros(1, device=device, dtype=dtype), horizons_t[:-1]])
        seg_hours = (horizons_t - prev_h)  # (H,)

        cur_lat = cur_pos_deg[:, 0:1]
        cur_lon = cur_pos_deg[:, 1:2]
        cos_lat = torch.cos(torch.deg2rad(cur_lat)).clamp_min(0.1)

        cum = torch.zeros(B, 2, device=device, dtype=dtype)  # (dx, dy) accumulated
        h_dec = h
        incs, logvars = [], []
        # use teacher forcing only if explicitly requested and truth is provided
        use_tf = teacher_forcing and (true_cum_km is not None)
        for i in range(H):
            # position feature from current state (predicted cumulative)
            lat = cur_lat + (cum[:, 1:2] / 111.32)
            lon = cur_lon + (cum[:, 0:1] / (111.32 * cos_lat))
            pos_feat = self.pos_mlp(torch.cat([lat, lon], dim=1))
            dt_feat = self.dt_mlp(seg_hours[i:i + 1].expand(B, 1))
            inp = torch.cat([h_dec, pos_feat, dt_feat], dim=1)
            h_dec = self.cell(self.drop(inp), h_dec)
            inc = self.out_inc(h_dec)           # (B, 2) segment increment km
            lv = self.out_logvar(h_dec).clamp(min=-10.0, max=5.0)
            new_cum = cum + inc
            incs.append(inc)
            logvars.append(lv)
            # next-step state: teacher-forcing "snaps" the state to the truth;
            # otherwise keep the accumulated prediction (autoregressive).
            if use_tf and i < H:
                cum = true_cum_km[:, i, :].clone() if i < true_cum_km.shape[1] else cum
            else:
                cum = new_cum
        incs = torch.stack(incs, dim=1)         # (B,H,2)
        logvars = torch.stack(logvars, dim=1)   # (B,H,2)
        cum_out = torch.cumsum(incs, dim=1)     # (B,H,2) cumulative km
        return cum_out, incs, logvars


class CycloneTransformerV5(nn.Module):
    """V5: same Transformer encoder as V3, configurable decoder.

    forward(x, cur_pos=None, true_cum_km=None, teacher_forcing=False) ->
        { "delta"   : (B,H,2) output in the active target space
          "log_var" : (B,H,2)
          "incs_km" : (B,H,2) per-segment km increments (sequential only)
          "cum_km"  : (B,H,2) cumulative local-km displacement (sequential only) }
    """

    def __init__(self, feature_dim, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, horizons=(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24),
                 target_mode="delta", decoder="direct"):
        super().__init__()
        self.horizons = list(horizons)
        self.target_mode = target_mode
        self.decoder = decoder
        num_horizons = len(self.horizons)

        self.input_mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        if decoder == "direct":
            self.decoder_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, num_horizons * 4),
            )
            self._seq = None
        elif decoder == "sequential":
            self._seq = _SeqDecoder(d_model, self.horizons, dim_feedforward, dropout)
            self.decoder_head = None
        else:
            raise ValueError(f"unknown decoder {decoder}")

    def forward(self, x, cur_pos=None, true_cum_km=None, teacher_forcing=False,
                src_key_padding_mask=None):
        h = self.input_mlp(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            lengths = (~src_key_padding_mask).sum(dim=1) - 1
            summary = h[torch.arange(h.size(0)), lengths]
        else:
            summary = h[:, -1, :]

        if self.decoder == "sequential":
            # cur_pos is required for the sequential decoder (raw anchor lat/lon).
            if cur_pos is None:
                raise ValueError("sequential decoder requires cur_pos (B,2) lat/lon")
            cum_km, incs, log_var = self._seq(summary, cur_pos, true_cum_km,
                                              teacher_forcing=teacher_forcing)
            # Convert cumulative local-km displacement into the active target space.
            delta = self._cum_km_to_target(cum_km, incs, cur_pos)
            return {"delta": delta, "log_var": log_var,
                    "incs_km": incs, "cum_km": cum_km}

        out = self.decoder_head(summary)            # (B, H*4)
        out = out.view(-1, len(self.horizons), 4)
        delta = out[..., 0:2]
        log_var = out[..., 2:4].clamp(min=-10.0, max=5.0)
        return {"delta": delta, "log_var": log_var}

    def _cum_km_to_target(self, cum_km, incs, cur_pos):
        """convert cumulative local-km output (from the sequential decoder) into
        the active target representation so the loss interprets `delta` uniformly."""
        cur_lat = cur_pos[:, 0:1]
        cos_lat = torch.cos(torch.deg2rad(cur_lat)).clamp_min(0.1)
        dx = cum_km[..., 0]
        dy = cum_km[..., 1]
        if self.target_mode == "km":
            return cum_km.clone()
        if self.target_mode == "delta":
            dlat = dy / 111.32
            dlon = dx / (111.32 * cos_lat)
            return torch.stack([dlat, dlon], dim=-1)
        if self.target_mode == "motion":
            # per-segment velocity from increments
            horizons_t = torch.as_tensor(list(self.horizons), dtype=dx.dtype, device=dx.device)
            prev_h = torch.cat([torch.zeros(1, device=dx.device, dtype=dx.dtype), horizons_t[:-1]])
            seg_hours = (horizons_t - prev_h).unsqueeze(0)
            u = incs[..., 0] / seg_hours
            v = incs[..., 1] / seg_hours
            return torch.stack([u, v], dim=-1)
        raise ValueError(f"unknown target_mode {self.target_mode}")


# ---------------------------------------------------------------------------
# V6 model. Same proven Transformer encoder (d_model=128, nhead=4, num_layers=3,
# dim_feedforward=256) and DIRECT, target-representation-agnostic decoder, but:
#   * each forecast horizon gets its own lightweight prediction head sharing the
#     Transformer context (no separate transformers, no sequential decoder);
#   * an optional LEARNED forecast-horizon embedding (3h/6h/12h/24h) is combined
#     with the shared context before each horizon's head, so the model explicitly
#     knows which lead time it is predicting;
#   * optional external `environment_features` interface (V6 sections 13-15) --
#     a future steering/environmental tensor can be fused in without restructuring
#     the model. When None (default for V6), the existing proxy features are used
#     and the model works exactly as before.
# Output: {"delta": (B, H, 2)} in the active target space (default "km":
#   delta[h] = (dx_km EAST, dy_km NORTH) cumulative displacement).
# ---------------------------------------------------------------------------
class CycloneTransformerV6(nn.Module):
    """V6 accuracy-focused model: shared encoder + horizon embedding + per-horizon
    direct heads. d_model stays 128 (spec #16) -- the added capacity comes from
    longer temporal context, horizon-aware heads and better features, not a larger
    Transformer."""

    def __init__(self, feature_dim, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, horizons=(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24),
                 target_mode="km", use_horizon_embedding=True,
                 env_feature_dim=None):
        super().__init__()
        self.horizons = list(horizons)
        self.target_mode = target_mode
        self.use_horizon_embedding = use_horizon_embedding
        H = len(self.horizons)

        # Shared numerical-feature encoder (identical scaffolding to V5/V3).
        self.input_mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Optional environmental-feature projection (V6 #15). Not required.
        self.env_proj = None
        if env_feature_dim is not None:
            self.env_proj = nn.Sequential(
                nn.Linear(env_feature_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

        # Learned per-horizon embedding (V6 #6). Disable for ablation B.
        if use_horizon_embedding:
            self.horizon_embeds = nn.Parameter(torch.randn(H, d_model) * (d_model ** -0.5))
        else:
            self.horizon_embeds = None

        # Lightweight per-horizon prediction heads (V6 #7). Each maps the shared
        # context (optionally concatenated with its horizon embedding) to the 2-D
        # target (dx_km EAST, dy_km NORTH).
        head_in = d_model * 2 if use_horizon_embedding else d_model
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 2),
            )
            for _ in range(H)
        ])

    def forward(self, x, cur_pos=None, environment_features=None,
                src_key_padding_mask=None):
        # x: (B, seq_len, feature_dim)
        h = self.input_mlp(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            lengths = (~src_key_padding_mask).sum(dim=1) - 1
            summary = h[torch.arange(h.size(0)), lengths]
        else:
            summary = h[:, -1, :]                       # (B, d_model) shared context

        # Optional external environmental fusion (V6 #15).
        if self.env_proj is not None and environment_features is not None:
            env = self.env_proj(environment_features)   # (B, d_model)
            summary = summary + env

        B = summary.shape[0]
        outs = []
        for i in range(len(self.horizons)):
            if self.use_horizon_embedding:
                embed = self.horizon_embeds[i].unsqueeze(0).expand(B, -1)
                combined = torch.cat([summary, embed], dim=1)  # (B, 2*d_model)
            else:
                combined = summary
            outs.append(self.heads[i](combined))        # (B, 2)
        out = torch.stack(outs, dim=1)                  # (B, H, 2)
        return {"delta": out, "log_var": None}


# ---------------------------------------------------------------------------
# V7 model: distance + direction (magnitude/direction) target representation.
#
# Same proven Transformer encoder as V5/V6 (d_model=128, nhead=4, num_layers=3,
# dim_feedforward=256) and a DIRECT, non-autoregressive decoder. The only change
# is the PREDICTION HEAD TARGET: instead of (dx_km, dy_km) east/north, each
# per-horizon head predicts the cyclone's movement MAGNITUDE and DIRECTION:
#
#   delta[h] = ( distance_km, direction_sin, direction_cos )
#
# where distance is the great-circle distance from the current position and
# (sin, cos) encode the great-circle initial bearing (0-360 clockwise from
# north), avoiding the 0/360 discontinuity. This lets the network learn
# magnitude and direction somewhat independently, targeting the known
# long-horizon underprediction.
#
# Architecture (spec #5):
#   historical sequence -> Transformer encoder -> shared context
#   -> horizon-conditioned direct heads -> distance + direction
#
# Horizon embeddings are OPTIONAL (--use_horizon_embedding, default OFF for the
# main V7 experiment) so the target representation can be evaluated alone.
# ---------------------------------------------------------------------------
class CycloneTransformerV7(nn.Module):
    """Shared encoder + direct per-horizon heads outputting
    (distance_km, direction_sin, direction_cos) per forecast horizon."""

    def __init__(self, feature_dim, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, horizons=(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24),
                 target_mode="distance_direction", use_horizon_embedding=False):
        super().__init__()
        self.horizons = list(horizons)
        self.target_mode = target_mode
        self.use_horizon_embedding = use_horizon_embedding
        H = len(self.horizons)
        self.out_dim = 3 if target_mode == "distance_direction" else 2

        # Shared numerical-feature encoder (identical scaffolding to V5/V6).
        self.input_mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Optional learned per-horizon embedding (spec #6).
        if use_horizon_embedding:
            self.horizon_embeds = nn.Parameter(torch.randn(H, d_model) * (d_model ** -0.5))
        else:
            self.horizon_embeds = None

        # Per-horizon direct heads -> (distance_km, direction_sin, direction_cos).
        head_in = d_model * 2 if use_horizon_embedding else d_model
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_in, d_model),
                nn.ReLU(),
                nn.Linear(d_model, self.out_dim),  # 3 (dist,sin,cos) or 2 (dx,dy)
            )
            for _ in range(H)
        ])

    def forward(self, x, cur_pos=None, src_key_padding_mask=None):
        # x: (B, seq_len, feature_dim)
        h = self.input_mlp(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            lengths = (~src_key_padding_mask).sum(dim=1) - 1
            summary = h[torch.arange(h.size(0)), lengths]
        else:
            summary = h[:, -1, :]                    # (B, d_model) shared context

        B = summary.shape[0]
        outs = []
        for i in range(len(self.horizons)):
            if self.use_horizon_embedding:
                embed = self.horizon_embeds[i].unsqueeze(0).expand(B, -1)
                combined = torch.cat([summary, embed], dim=1)   # (B, 2*d_model)
            else:
                combined = summary
            outs.append(self.heads[i](combined))                 # (B, 3)
        out = torch.stack(outs, dim=1)                           # (B, H, 3)
        return {"delta": out, "log_var": None}


# ---------------------------------------------------------------------------
# V9 model: V5-preserving magnitude calibration + direction supervision.
#
# The V8 diagnostic identified the PRIMARY failure as systematic long-horizon
# displacement UNDERPREDICTION (24h median ratio ~0.958) with DIRECTION as a
# secondary contributor (24h mean |bearing error| ~19.9 deg). V7's separate
# distance_direction target regressed, so V9 deliberately KEEPS the winning
# V5 (dx_km, dy_km) primary target and decoder, and instead attaches three
# LIGHTWEIGHT auxiliary heads to the SAME shared Transformer context:
#
#   * magnitude head : predicts normalized log1p(distance) per horizon, so the
#                      network is explicitly supervised about movement magnitude.
#   * direction head : predicts (direction_sin, direction_cos) = sin/cos of the
#                      east/north bearing atan2(dx, dy) per horizon, giving the
#                      network explicit direction supervision without adopting
#                      V7's representation (which lost the proven dx/dy path).
#   * scale head     : predicts a horizon-specific BOUNDED correction factor
#                      scale = 1 + max_correction * tanh(raw_scale), applied to
#                      the raw dx/dy via
#                          dx_calibrated = dx_raw * scale
#                          dy_calibrated = dy_raw * scale
#                      so ONLY the movement magnitude is rescaled (direction is
#                      untouched) and the correction is learned from training
#                      data -- never a hand-picked test-set multiplier.
#
# Architecture stays small and is identical to V5 up to the shared context:
#   historical sequence -> Transformer encoder -> shared context
#     -> [dx/dy head] [magnitude head] [direction head] [scale head]
#     -> learned magnitude correction -> calibrated dx/dy
#     -> latitude/longitude -> Haversine evaluation
#
# The primary dx/dy output remains (B, H, 2) in the SAME local-kilometer
# east/north space as V5 target_mode="km", so V5 and V9 are directly comparable
# and V9 can be scored with reconstruct_positions(..., "km").
# ---------------------------------------------------------------------------
class CycloneTransformerV9(nn.Module):
    """V5-identical shared encoder + direct dx/dy decoder, plus lightweight
    magnitude / direction / calibration heads sharing the same context.

    forward(x, cur_pos=None, src_key_padding_mask=None) -> dict with:
        "delta"              : (B,H,2) RAW dx_km (east), dy_km (north)
        "delta_calibrated"   : (B,H,2) calibrated = delta * scale
        "log_var"            : (B,H,2) per-component log-variance (on raw delta)
        "magnitude_pred"     : (B,H,1) predicted normalized log1p(distance)
        "direction_pred"     : (B,H,2) predicted (sin, cos) of east/north bearing
        "scale"              : (B,H,1) learned bounded correction near 1
    """

    def __init__(self, feature_dim, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, horizons=(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24),
                 target_mode="km", max_magnitude_correction=0.15):
        super().__init__()
        self.horizons = list(horizons)
        self.target_mode = target_mode
        self.max_magnitude_correction = max_magnitude_correction
        H = len(self.horizons)
        assert target_mode == "km", "V9 primary target must be 'km' (dx/dy)"

        # Shared numerical-feature encoder (identical to V5/V6/V7).
        self.input_mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Primary direct dx/dy decoder (matches V5 decoder="direct"): each
        # horizon -> (dx, dy, log_var_x, log_var_y).
        self.decoder_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, H * 4),
        )

        # Lightweight auxiliary heads from the same shared context.
        # magnitude: raw log1p-distance -> normalized output (1 value per horizon).
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, H),          # (B, H) raw log-distance per horizon
        )
        # direction: (sin, cos) per horizon.
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, H * 2),      # (B, H*2) -> (B, H, 2)
        )
        # scale: raw scalar per horizon, bounded via tanh in forward.
        self.scale_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, H),          # (B, H) raw scale per horizon
        )

    def forward(self, x, cur_pos=None, src_key_padding_mask=None):
        h = self.input_mlp(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            lengths = (~src_key_padding_mask).sum(dim=1) - 1
            summary = h[torch.arange(h.size(0)), lengths]
        else:
            summary = h[:, -1, :]                       # (B, d_model)

        H = len(self.horizons)

        # Primary dx/dy (raw) + log-variance.
        out = self.decoder_head(summary)                # (B, H*4)
        out = out.view(-1, H, 4)
        delta = out[..., 0:2]
        log_var = out[..., 2:4].clamp(min=-10.0, max=5.0)

        # Learned bounded magnitude scale, close to 1: 0.85..1.15 by default.
        raw_scale = self.scale_head(summary)            # (B, H)
        scale = 1.0 + self.max_magnitude_correction * torch.tanh(raw_scale)
        scale = scale.unsqueeze(-1)                     # (B, H, 1)

        delta_calibrated = delta * scale                # magnitude-only rescale

        # Magnitude head (normalized log-distance) and direction head.
        mag_raw = self.magnitude_head(summary)          # (B, H)
        magnitude_pred = mag_raw.unsqueeze(-1)          # (B, H, 1)
        dir_out = self.direction_head(summary)          # (B, H*2)
        direction_pred = dir_out.view(-1, H, 2)         # (B, H, 2)

        return {
            "delta": delta,
            "delta_calibrated": delta_calibrated,
            "log_var": log_var,
            "magnitude_pred": magnitude_pred,
            "direction_pred": direction_pred,
            "scale": scale,
        }


# ---------------------------------------------------------------------------
# V11 model: IDENTIFIABLE magnitude calibration.
#
# V9/V10 demonstrated that a free learned scale head and the raw dx/dy head can
# trade off against each other: V10's raw head collapsed to ~0.48x and the scale
# head saturated at 2.0 to compensate, so the calibration was meaningless.
#
# V11 makes the calibration IDENTIFIABLE by construction:
#
#   1. KEEP the V5 primary (dx_km, dy_km) direct decoder untouched.
#   2. Attach a SMALL, bounded, horizon-specific scale head:
#          scale_h = 1 + max_correction * tanh(raw_scale_h),  max_correction=0.20
#      so 0.80 <= scale <= 1.20. Only magnitude is rescaled (direction invariant).
#   3. Initialize the scale head to output EXACTLY 1.0 for every horizon
#      (zero weight AND zero bias on the final layer => raw_scale = 0 => scale=1).
#      The model must LEARN any correction; nothing is pre-seeded from V8/V9/V10.
#   4. The RAW dx/dy head is anchored by a RAW-magnitude supervision loss so it
#      cannot shrink and let the scale compensate (see losses.v11_loss). Because
#      the anchoring constrains raw |(dx,dy)|, raw_ratio stays near 1 and scale
#      only makes a modest correction -- restoring identifiability.
#
# The Transformer encoder is IDENTICAL to V5/V9 (d_model=128, nhead=4,
# num_layers=3, dim_feedforward=256). Architecture is unchanged; only the
# auxiliary calibration head + its initialization differ.
#
# forward(x, cur_pos=None, src_key_padding_mask=None) -> dict:
#     "delta"            : (B,H,2) RAW dx_km (east), dy_km (north)
#     "delta_calibrated" : (B,H,2) = delta * scale  (final prediction)
#     "log_var"          : (B,H,2) per-component log-variance (on raw delta)
#     "magnitude_pred"   : (B,H,1) predicted normalized log1p(distance) [aux]
#     "direction_pred"   : (B,H,2) predicted (sin, cos) of bearing [disabled by default]
#     "scale"            : (B,H,1) learned bounded per-horizon correction in [0.8,1.2]
# ---------------------------------------------------------------------------
class CycloneTransformerV11(nn.Module):
    """V5 encoder + direct dx/dy decoder + identifiable bounded scale correction.

    The scale head starts at exactly 1.0 per horizon and is allowed to make only
    a modest learned correction around 1.0, while the raw dx/dy head is anchored
    so the two cannot compensate for each other (raw-head collapse).
    """

    def __init__(self, feature_dim, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1, horizons=(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24),
                 target_mode="km", max_magnitude_correction=0.20):
        super().__init__()
        self.horizons = list(horizons)
        self.target_mode = target_mode
        self.max_magnitude_correction = max_magnitude_correction
        H = len(self.horizons)
        assert target_mode == "km", "V11 primary target must be 'km' (dx/dy)"

        # Shared numerical-feature encoder (identical to V5/V6/V7/V9).
        self.input_mlp = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Primary direct dx/dy decoder (matches V5 decoder="direct"): each
        # horizon -> (dx, dy, log_var_x, log_var_y). UNCHANGED from V5.
        self.decoder_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, H * 4),
        )

        # Lightweight magnitude / direction auxiliary heads (kept from V9, but
        # their loss contributions default to 0.0 in the V11 loss).
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, H),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, H * 2),
        )

        # ---- Identifiable scale head (V11) ----
        # Hidden -> then the FINAL layer maps to one scalar per horizon. We
        # initialize the final layer's weight to zero AND its bias to zero so
        # raw_scale == 0 for every horizon at init => scale == 1.0 exactly.
        # Starting from scale=1 (not 1.1 or any measured value) is required so
        # the model must learn any correction from the data.
        self.scale_head_hidden = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )
        self.scale_head_out = nn.Linear(d_model, H)      # (B, H) raw scale
        nn.init.zeros_(self.scale_head_out.weight)
        nn.init.zeros_(self.scale_head_out.bias)

    def forward(self, x, cur_pos=None, src_key_padding_mask=None):
        h = self.input_mlp(x)
        h = self.pos_enc(h)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            lengths = (~src_key_padding_mask).sum(dim=1) - 1
            summary = h[torch.arange(h.size(0)), lengths]
        else:
            summary = h[:, -1, :]                       # (B, d_model)

        H = len(self.horizons)

        # Primary dx/dy (raw) + log-variance.
        out = self.decoder_head(summary)                # (B, H*4)
        out = out.view(-1, H, 4)
        delta = out[..., 0:2]
        log_var = out[..., 2:4].clamp(min=-10.0, max=5.0)

        # Identifiable bounded scale near 1: 0.80..1.20 by default.
        hid = self.scale_head_hidden(summary)           # (B, d_model)
        raw_scale = self.scale_head_out(hid)            # (B, H)
        scale = 1.0 + self.max_magnitude_correction * torch.tanh(raw_scale)
        scale = scale.unsqueeze(-1)                     # (B, H, 1)

        delta_calibrated = delta * scale                # magnitude-only rescale

        # Aux heads (kept; direction loss defaults off in V11).
        magnitude_pred = self.magnitude_head(summary).unsqueeze(-1)  # (B, H, 1)
        dir_out = self.direction_head(summary)          # (B, H*2)
        direction_pred = dir_out.view(-1, H, 2)         # (B, H, 2)

        return {
            "delta": delta,
            "delta_calibrated": delta_calibrated,
            "log_var": log_var,
            "magnitude_pred": magnitude_pred,
            "direction_pred": direction_pred,
            "scale": scale,
        }


if __name__ == "__main__":
    # quick smoke test
    model = CycloneTransformer(feature_dim=FEATURE_DIM, horizons=(1, 2, 4, 8))
    x = torch.randn(5, 12, FEATURE_DIM)
    out = model(x)
    print("delta:", out["delta"].shape)      # (5, 4, 2)
    print("log_var:", out["log_var"].shape)  # (5, 4, 2)

    v5 = CycloneTransformerV5(feature_dim=FEATURE_DIM, horizons=(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24),
                              decoder="sequential")
    pos = torch.randn(5, 2)
    out = v5(x, cur_pos=pos)
    print("v5 sequential delta:", out["delta"].shape, "log_var:", out["log_var"].shape)
    print("v5 cum_km:", out["cum_km"].shape, "incs_km:", out["incs_km"].shape)
