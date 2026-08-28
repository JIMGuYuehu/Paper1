# Methods implemented in code

- O3: integrate 30-70 hPa with exact boundary interpolation, then calculate the
  60-90 N cosine-weighted mean. Spring minima use centred 5-day means for
  1 March-30 April. The fixed WACCM threshold is ranked from 207 LONGRUN +
  23 BWCN springs.
- NAM/AO: train fixed monthly EOFs; use a no-leap calendar-day climatology with
  a centred 21-day smoother; AO is the 1000 hPa NAM level.
- EP flux: Jucker pressure-coordinate implementation with `do_ubar=True`,
  `w=None`, all resolved waves (`wave=-1`), natural-calendar-month static
  stability, upward flux `Fz_upward=-ep2`, and a 40-80 N cosine mean. Restart
  predictors remain in physical units; multi-spring anomalies are standardized
  by pressure and calendar day within each dataset.
- Chemistry: MLS daytime ClO/H2O and WACCM ClOx/H2O anomalies over 60-82 N and
  1-100 hPa. Figure 3 retains only the 195 K PSC-I threshold/mask.
- Hindcasts: 30-member population spread (`ddof=0`), finite-member CRPS,
  memberwise RMSE after exact date matching, and at least 90% sign agreement.
  All reported relationships use two-sided Pearson correlation and stored OLS
  display fits; no Spearman branch is present.
- Figure 15: stationary Z300 is the monthly anomaly minus its zonal mean.
  Event and same-sized low-O3 composites use 5,000 bootstrap resamples.

Known manuscript-asset issues as of 2026-08-28:

- Figure 1a is reproduced from the checksum-gated accepted PNG because the
  exact legacy envelope input pools have not been recovered.
- The current Figure 8h PDF contains the legacy BWCN reference line; the
  released code uses the no-W-correction reference. The regression itself was
  already no-W-correction.
- The current Figure 9b PDF is the old 20 February-13 March (22-day) asset.
  The released code and manuscript text use 21 February-12 March (20 days), so
  the PDF must be regenerated before final submission.
- The current Figure 6b caption lists O3, EP flux and U60N10 but omits the
  fourth Tmin50 panel; the plotted data and released code both include Tmin50.
