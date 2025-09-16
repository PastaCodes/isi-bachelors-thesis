OBSERVATORY_FORMAT: Final[str] = '(A3, 1X, F9.5, F8.6, F9.6, A50)'
observatory_reader = FortranRecordReader(OBSERVATORY_FORMAT)

def load_observatories():
    with open(PROJECT_ROOT + observatories_file_path, 'rt', encoding='utf-8') as file:
        for line in file.readlines()[1:]:
            if not line[6]:  # No location data; likely not on Earth
                continue
            code, lon_deg, p1, p2, name = observatory_reader.read(line)
            loc = location_from_parallax(float(lon_deg), float(p1), float(p2))
            observatories[code] = Observatory(code, name, loc)


OBSERVATION_FORMAT: Final[str] = '(A5, A7, A1, A1, A1, I4, 1X, I2, 1X, F9.6, I2, 1X, I2, 1X, F6.3, I3, 1X, I2, 1X, F5.2, 9X, F5.2, A1, 6X, A3)'
observation_reader = FortranRecordReader(OBSERVATION_FORMAT)

def parse_observation_line(body: MinorPlanet, line: str) -> MinorPlanetObservation:
    _packed_num, _packed_desig, _discov_ast, _note1, _note2, year, month, day_decimal, ra_hour, ra_minute, ra_second, dec_degree, dec_minute, dec_second, mag, band, observatory_code = observation_reader.read(line)
    observatory = observatories[observatory_code]
    epoch = decimal_day_date_to_time(year, month, day_decimal, observatory.location)
    return MinorPlanetObservation.with_band(target_body=body, epoch=epoch, observatory=observatory, ra=np.radians(15 * ra_hour + ra_minute / 4 + ra_second / 240), dec=np.radians(dec_degree + dec_minute / 60 + dec_second / 3600), observed_magnitude=mag, band=band)


def parse_observations(body: MinorPlanet,
                       accept_methods: Collection[str] | None = ('B', 'C')) -> Iterable[MinorPlanetObservation]:
    if not observatories:
        load_observatories()
    with open(PROJECT_ROOT + body.observations_filepath, 'rt', encoding='utf-8') as file:
        prev_time = None
        for line in file.readlines():
            if not line[65:71].strip():  # No magnitude
                continue
            if accept_methods is not None:
                if line[14] not in accept_methods:
                    continue
            else:
                if not line[14].isupper():
                    continue
            obs = parse_observation_line(body, line)
            if obs.epoch == prev_time:
                continue
            yield obs
            prev_time = obs.epoch


def parse_ephemeris(body: MinorPlanet) -> Iterable[MinorPlanetEphemeris]:
    with open(PROJECT_ROOT + body.ephemeris_filepath, 'rt', encoding='utf-8') as file:
        for line in file.readlines():
            (t, tx, ty, tz, ox, oy, oz, sx, sy, sz, a, e, i, om, w, mm, v, n, astro_ra, astro_dec,
             app_ra, app_dec, app_vv, app_th, app_phi, astro_phi) = line.split(' ')
            yield MinorPlanetEphemeris(body, Time(float(t), format='jd', scale='tdb'),
                                       np.array([float(tx), float(ty), float(tz)]),
                                       np.array([float(ox), float(oy), float(oz)]),
                                       np.array([float(sx), float(sy), float(sz)]),
                                       float(a), float(e), np.radians(float(i)), np.radians(float(om)),
                                       np.radians(float(w)), np.radians(float(mm)), np.radians(float(v)),
                                       np.radians(float(n)), np.radians(float(astro_ra)), np.radians(float(astro_dec)),
                                       np.radians(float(app_ra)), np.radians(float(app_dec)), float(app_vv),
                                       np.radians(float(app_th)), np.radians(float(app_phi)),
                                       np.radians(float(astro_phi)))
