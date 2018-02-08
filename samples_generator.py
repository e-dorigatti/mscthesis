from layered_atmosphere import (RandomAtmosphere, RandomNoisyAtmosphere, RandomLayer,
                                UniformSampler, GaussianSamplerBetween, ConstantSampler,
                                random_noise)
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
        GaussianSamplerBetween(50, 250),
        GaussianSamplerBetween(1, 3),
        GaussianSamplerBetween(50, 150),
        GaussianSamplerBetween(25, 75),
    )



def random_relative_humidity(samples_x=None, num_samples=None, pmin=10, pmax=100000,
                             num_support_vectors=10, svr_c=2, svr_gamma_norm=100):

    fn_x, fn_y, _ = random_noise(samples_x, num_samples, num_support_vectors,
                                 svr_c, svr_gamma_norm)

    # scale random noise so that it shows a decreasing trend
    # with  0% RH at the top of the atmosphere
    of = 0.2 +  np.random.random() * 0.3
    btop = 1 / (1 + np.exp(-(fn_x - of) * 25))
    bbot = fn_x / pmax * 0.1

    fn_x = pmin + fn_x * (pmax - pmin)
    fn_y = (fn_y + 1) / 2

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
    _, relhum = random_relative_humidity(pmin=min_ps, pmax=max_ps, samples_x=press)
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
