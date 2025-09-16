def real_position(t: float, a: float, e: float, i: float, om: float, w: float,
                  n: float, tp: float) -> np.ndarray:
    mm = wrap_angle(n * (t - tp))
    ee = eccentric_anomaly_from_mean_anomaly(mm, e)
    v = true_anomaly_from_eccentric_anomaly(ee, e)
    r = distance_from_eccentric_anomaly(ee, a, e)
    return position_from_orbital_angles(om, w + v, i, r)


def observe(tgt_pos: np.ndarray, obs_pos: np.ndarray, sun_pos: np.ndarray,
            hh: float, gg: float, dir_var: float, vv_var: float,
            rng: np.random.Generator) -> np.ndarray:
    tgt_obs_pos = tgt_pos - obs_pos
    tgt_obs_dist = norm(tgt_obs_pos)
    tgt_obs_ray = tgt_obs_pos / tgt_obs_dist
    tgt_obs_ray = sp.stats.vonmises_fisher(mu=tgt_obs_ray, kappa=(1.0 / dir_var), seed=rng).rvs(1)[0]
    ra, dec = direction_to_ra_dec(tgt_obs_ray)
    tgt_sun_dist = norm(tgt_pos - sun_pos)
    obs_sun_dist = norm(obs_pos - sun_pos)
    phi = phase_from_distances(tgt_sun_dist, tgt_obs_dist, obs_sun_dist)
    vv = visual_magnitude_from_absolute(hh, tgt_sun_dist, tgt_obs_dist, phi, gg)
    vv = rng.normal(loc=vv, scale=np.sqrt(vv_var))
    return np.array([ra, dec, vv])


def main() -> None:
    # ...
    points = MerweScaledSigmaPoints(13, alpha=1E-3, beta=2, kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=13, dim_z=5, dt=None, hx=measure, fx=propagate, points=points)

    x0 = initial_state(prepared[0], obs_pos[0], sun_pos[0], a0, e0, i0, n0, hh0, gg0, mm0_hint, om0_hint)
    pp0 = state_autocovariance_matrix(x0, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3, 1E-3)
    prev_t = times[0]
    ukf.x = x0
    ukf.P = pp0

    estimates = [finalize_transform(x0)]
    for t, obs, o, s in zip(times[1:], prepared[1:], obs_pos[1:], sun_pos[1:]):
        dt = t - prev_t
        ukf.Q = state_autocovariance_matrix(ukf.x, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8, 1E-8)
        ukf.predict(dt=dt)
        ukf.R = measurement_autocovariance_matrix(obs, dir_var, vv_var)
        ukf.update(z=obs, obs_pos=o, sun_pos=s)
        estimates.append(finalize_transform(ukf.x))
        prev_t = t
    # ...
