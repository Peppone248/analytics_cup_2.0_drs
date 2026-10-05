"""Ghost — belief-state awareness model on SkillCorner body-pose data.

Port of the master-thesis Track A pipeline (FoV weights w_ij -> adaptive-R
Kalman belief state) to the SkillCorner open body-pose matches, with a new
awareness readout based on the "ghost": the position where the defender
*believes* the attacker is (Kalman mean) vs. where the attacker really is.

Modules
-------
data         : stream pose archives -> compact per-frame player/ball tables
orientation  : head / shoulder orientation, gating, smoothing, gap filling
fov          : field-of-view + occlusion weights w_ij (Bekkers SSAC26, as in thesis)
kalman       : batched adaptive-R constant-velocity Kalman filter (thesis)
scores       : awareness readouts (covariance-based, ghost-error, Mahalanobis)
pipeline     : one window -> long (frame, observer, target) dataframe
viz2d        : 2D pitch renders (static + animated) with ghosts and FoV cones
export3d     : JSON export for the three.js POV viewer (viewer/pov3d.html)
"""
