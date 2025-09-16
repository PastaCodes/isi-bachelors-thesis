def reverse_transform(obs_vec: np.ndarray, obs_pos: np.ndarray, sun_pos: np.ndarray,
                      hh: float, gg: float) -> np.ndarray:
    cos_ra, sin_ra, cos_dec, sin_dec, vv = obs_vec
    tgt_obs_ray = ra_dec_components_to_direction(cos_ra, sin_ra, cos_dec, sin_dec)
    sun_obs_pos = sun_pos - obs_pos
    sun_obs_dist = norm(sun_obs_pos)
    sun_obs_ray = sun_obs_pos / sun_obs_dist
    th = elongation_from_directions(tgt_obs_ray, sun_obs_ray)
    tgt_obs_dist, _ = solve_distance_and_phase(vv, sun_obs_dist, th, hh, gg)
    tgt_obs_pos = tgt_obs_dist * tgt_obs_ray
    tgt_pos = tgt_obs_pos + obs_pos
    return tgt_pos


def initial_state(obs_vec: np.ndarray, obs_pos: np.ndarray, sun_pos: np.ndarray,
                  a: float, e: float, i: float, n: float, hh: float, gg: float,
                  mm_hint: float, om_hint: float) -> np.ndarray:
    tgt_pos = reverse_transform(obs_vec, obs_pos, sun_pos, hh, gg)
    r = norm(tgt_pos)
    ee = eccentric_anomaly_from_distance(r, a, e, mm_hint)
    mm = mean_anomaly_from_eccentric_anomaly(ee, e)
    v = true_anomaly_from_eccentric_anomaly(ee, e)
    om, w = orbital_angles_from_position(tgt_pos, v, i, om_hint)
    return np.array([*angle_components(mm), a, e, *angle_components(i),
                     *angle_components(om), *angle_components(w), n, hh, gg])


def finalize_transform(state_vec: np.ndarray) -> np.ndarray:
    cos_mm, sin_mm, a, e, cos_i, sin_i, cos_om, sin_om, cos_w, sin_w, n, hh, gg = state_vec
    mm = angle_from_components(cos_mm, sin_mm)
    ee = eccentric_anomaly_from_mean_anomaly(mm, e)
    v = true_anomaly_from_eccentric_anomaly(ee, e)
    cos_v, sin_v = angle_components(v)
    r = distance_from_eccentric_anomaly(ee, a, e)
    return position_from_orbital_angle_components(cos_om, sin_om, cos_w, sin_w, cos_v, sin_v, cos_i, sin_i, r)


def measure(state_vec: np.ndarray, obs_pos: np.ndarray,
            sun_pos: np.ndarray) -> np.ndarray:
    tgt_pos = finalize_transform(state_vec)
    hh, gg = state_vec[11], state_vec[12]
    tgt_obs_pos = tgt_pos - obs_pos
    tgt_obs_dist = norm(tgt_obs_pos)
    tgt_obs_ray = tgt_obs_pos / tgt_obs_dist
    cos_ra, sin_ra, cos_dec, sin_dec = direction_to_ra_dec_components(tgt_obs_ray)
    tgt_sun_dist = norm(tgt_pos - sun_pos)
    obs_sun_dist = norm(obs_pos - sun_pos)
    phi = phase_from_distances(tgt_sun_dist, tgt_obs_dist, obs_sun_dist)
    vv = visual_magnitude_from_absolute(hh, tgt_sun_dist, tgt_obs_dist, phi, gg)
    return np.array([cos_ra, sin_ra, cos_dec, sin_dec, vv])


def propagate(before_vec: np.ndarray, dt: float) -> np.ndarray:
    cos_mm_before, sin_mm_before, a, e, cos_i, sin_i, cos_om, sin_om, cos_w, sin_w, n, hh, gg = before_vec
    mm_before = angle_from_components(cos_mm_before, sin_mm_before)
    mm_after = advance_mean_anomaly(mm_before, n, dt)
    return np.array([*angle_components(mm_after), a, e, cos_i, sin_i, cos_om, sin_om, cos_w, sin_w, n, hh, gg])


def angle_components_autocovariance_matrix(cos: float, sin: float, var: float) -> np.ndarray:
    mat = var * np.array([[sin * sin, -sin * cos],
                          [-sin * cos, cos * cos]])
    return mat


def state_autocovariance_matrix(state_vec: np.ndarray, mm_var: float, a_var: float, e_var: float, i_var: float,
                                om_var: float, w_var: float, n_var: float, hh_var: float, gg_var: float) -> np.ndarray:
    cos_mm, sin_mm, _, _, cos_i, sin_i, cos_om, sin_om, cos_w, sin_w, _, _, _ = state_vec
    qq = np.zeros((13, 13))
    qq[0:2, 0:2] = angle_components_autocovariance_matrix(cos_mm, sin_mm, mm_var)
    qq[2, 2], qq[3, 3] = a_var, e_var
    qq[4:6, 4:6] = angle_components_autocovariance_matrix(cos_i, sin_i, i_var)
    qq[6:8, 6:8] = angle_components_autocovariance_matrix(cos_om, sin_om, om_var)
    qq[8:10, 8:10] = angle_components_autocovariance_matrix(cos_w, sin_w, w_var)
    qq[10, 10], qq[11, 11], qq[12, 12] = n_var, hh_var, gg_var
    return qq


def measurement_autocovariance_matrix(obs_vec: np.ndarray, dir_var: float, vv_var: float) -> np.ndarray:
    cos_ra, sin_ra, cos_dec, sin_dec, _ = obs_vec
    rr = np.zeros((5, 5))
    rr[0:2, 0:2] = angle_components_autocovariance_matrix(cos_ra / cos_dec, sin_ra / cos_dec, dir_var)
    rr[2:4, 2:4] = angle_components_autocovariance_matrix(cos_dec, sin_dec, dir_var)
    rr[4, 4] = vv_var
    return rr


def naive_transform(obs: MinorPlanetObservation, eph: MinorPlanetEphemeris) -> np.ndarray:
    return reverse_transform(obs.to_vector(), eph.observer_position, eph.sun_position,
                             obs.target_body.hh0, obs.target_body.gg0)
