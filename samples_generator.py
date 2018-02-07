from layered_atmosphere import (RandomAtmosphere, RandomNoisyAtmosphere, RandomLayer,
                                UniformSampler, GaussianSamplerBetween, ConstantSampler)
from sklearn.svm import SVR
import numpy as np


def default_random_atmosphere():
    gas_const = ConstantSampler(0.28704)
    return RandomNoisyAtmosphere(
        RandomAtmosphere(
            UniformSampler(273 - 100, 273 + 60),
            UniformSampler(70000, 110000),
            RandomLayer(UniformSampler(5, 25),  UniformSampler(-10, -0.1), gas_const),
            RandomLayer(UniformSampler(0.1, 3), UniformSampler(-1, 1), gas_const),
            RandomLayer(UniformSampler(8, 16),  UniformSampler(1, 2), gas_const),
            RandomLayer(UniformSampler(11, 19), UniformSampler(2, 3), gas_const),
            RandomLayer(UniformSampler(1, 7),   UniformSampler(-1, 1), gas_const),
            RandomLayer(UniformSampler(15, 25), UniformSampler(-5, -0.1), gas_const),
            RandomLayer(UniformSampler(10, 20), UniformSampler(-5, -0.1), gas_const),
        ),
        GaussianSamplerBetween(5, 25),
        GaussianSamplerBetween(-5, 5),
        GaussianSamplerBetween(1, 5),
    )


def random_relative_humidity(pmin, pmax, num_support_vectors=10, num_samples=100,
                             samples_x=None, svr_c=2, svr_gamma=1e-8):

    # basically generate random points and fit an SVR to them, then predict some samples
    # and scale the output so that it shows a decreasing trend, with  0% RH at the top of the atmosphere

    pts_x = np.array([pmin] + sorted(
        pmin + np.random.random(size=num_support_vectors) * (pmax - pmin)
    ) + [pmax])
    pts_y = np.random.random(size=len(pts_x))

    if samples_x is None:
        fn_x = np.arange(pmin, pmax, (pmax - pmin) / num_samples)
    else:
        fn_x = samples_x

    sx = (fn_x - fn_x.min()) / (fn_x.max() - fn_x.min())
    of = 0.2 +  np.random.random() * 0.3
    btop = 1 / (1 + np.exp(-(sx - of) * 25))

    bbot = fn_x / pmax * 0.1

    svr = SVR(C=svr_c, gamma=svr_gamma).fit(pts_x.reshape((-1, 1)), pts_y)
    fn_y = svr.predict(fn_x.reshape((-1, 1)))
    fn_y = np.where(fn_y < 0.01, 0.01, fn_y)
    fn_y = np.where(fn_y > 0.99, 0.99, fn_y)
    fn_y = bbot + fn_y * (btop - bbot)

    return fn_x, fn_y


def compute_absolute_humidity(relative_humidity, temperature, pressure):
    tempC = temperature - 273.15  # convert to deg C

    # buck equation, result in Pa
    saturation_press = 1000 * 0.61121 * np.exp((18.678 - tempC / 234.5) * (tempC / (257.14 + tempC)))

    # eq. 3.63 p 82
    abs_humidity = 0.622 * relative_humidity * saturation_press / pressure

    return np.where(abs_humidity < 0, 0, abs_humidity)


def make_sample(random_atmosphere=None, pressure_levels=None):
    rnd = random_atmosphere or default_random_atmosphere()
    ratm = rnd.sample()

    if not pressure_levels:
        max_ps = ratm.get_pressure_by_altitude(0.1) * 0.9999
        min_ps = ratm.get_pressure_by_altitude(ratm.height) * 1.0001
        press = np.arange(min_ps, max_ps, (max_ps - min_ps) / 100)
    else:
        max_ps, min_ps = max(pressure_levels), min(pressure_levels)
        press = pressure_levels

    temps = np.array([ratm.get_temperature_by_pressure(p) for p in press])
    _, relhum = random_relative_humidity(min_ps, max_ps, samples_x=press)
    abshum = compute_absolute_humidity(relhum, temps, press)

    return press, temps, abshum, relhum, ratm


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    press, temps, abshum, relhum, ratm = make_sample()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(temps, press)
    plt.ylabel('Pressure (Pa)')
    plt.xlabel('Temperature (K)')
    plt.gca().invert_yaxis()
    plt.subplot(1, 3, 2)
    plt.plot(abshum, press)
    plt.gca().invert_yaxis()
    plt.xlabel('Abs. humidity (g/g)')
    plt.subplot(1, 3, 3)
    plt.plot(relhum, press)
    plt.gca().invert_yaxis()
    plt.xlabel('Rel. humidity')
    plt.tight_layout()
    plt.show()
