import math
from sklearn.svm import SVR
from scipy.stats import norm
import numpy as np


class AtmosphericLayer:
    ''' in a layer, temperature changes linearly with temperature
    '''

    grav_const = 9.8066

    def __init__(self, height, lapse_rate, base_temp, base_press, gas_const=0.28704):
        if base_temp < 0:
            raise ValueError('bottom temperature cannot be below 0 deg. kelvin')

        if base_press <= 0:
            raise ValueError('pressure must be non-negative')

        self.height = height
        self.tbot = base_temp
        self.base_press = base_press
        self.gas_const = gas_const
        self.ttop = base_temp + lapse_rate * height

        if self.ttop < 0:
            raise ValueError('top temperature cannot be below 0 deg. kelvin')

    @property
    def lapse_rate(self):
        return (self.ttop - self.tbot) / self.height

    def get_temperature_by_altitude(self, altitude):
        # allow some slack due to numerical errors when computing alt. from pressure
        slack = 1e-10
        #assert -slack <= altitude <= self.height + slack, (altitude, self.height)
        return self.tbot + (self.ttop - self.tbot) * (altitude / self.height)

    def get_pressure_by_altitude(self, altitude):
        exp = -self.grav_const  * self.tbot * self.height / (self.gas_const * self.ttop)
        frac = 1 + altitude * self.ttop / (self.tbot**2 * self.height)
        return self.base_press * math.pow(frac, exp)

    def get_altitude_by_pressure(self, pressure):
        co = self.tbot**2 * self.height / self.ttop
        exp = self.gas_const * self.ttop / (self.grav_const * self.tbot * self.height)
        return co * (math.pow(self.base_press / pressure, exp) - 1)

    def get_temperature_by_pressure(self, pressure):
        altitude = self.get_altitude_by_pressure(pressure)
        return self.get_temperature_by_altitude(altitude)


class LayeredAtmosphere:
    ''' atmospheric models made by a sequence of layers
        inside of which temperature changes linearly with altitude
    '''
    def __init__(self, layers=None):
        self.layers = layers or []

    @staticmethod
    def international_standard_atmosphere():
        # https://en.wikipedia.org/wiki/International_Standard_Atmosphere
        atm = LayeredAtmosphere()
        atm.add_layer(11.019, -6.5, 273.15 + 19.0, 101325)  # troposphere
        atm.add_layer(20.063 - 11.019, 0.0)                 # tropopause
        atm.add_layer(32.162 - 20.063, 1.0)                 # stratosphere
        atm.add_layer(47.350 - 32.162, 2.8)                 # stratosphere
        atm.add_layer(51.413 - 47.350, 0.0)                 # stratopause
        atm.add_layer(71.802 - 51.413, -2.8)                # mesosphere
        atm.add_layer(84.852 - 71.802, -2.0)                # mesosphere
        return atm

    @property
    def height(self):
        return sum(l.height for l in self.layers)

    def add_layer(self, height, lapse_rate, base_temp=None, base_press=None, gas_const=0.2874):
        base_temp = base_temp or self.layers[-1].get_temperature_by_altitude(self.layers[-1].height)
        base_press = base_press or self.layers[-1].get_pressure_by_altitude(self.layers[-1].height)

        self.layers.append(AtmosphericLayer(height, lapse_rate, base_temp, base_press, gas_const))

    def _get_layer_by_altitude(self, altitude):
        alt = 0.0
        for layer in self.layers:
            alt += layer.height
            if altitude <= alt:
                return layer, altitude - alt + layer.height
        else:
            raise ValueError('altitude %f is outside of the atmosphere, limit is %f' % (
                    altitude, alt))

    def _get_layer_by_pressure(self, pressure):
        for layer in self.layers:
            if pressure >= layer.get_pressure_by_altitude(layer.height):
                return layer
        else:
            raise ValueError('pressure outside of atmosphere: %f' % pressure)

    def get_temperature_by_altitude(self, altitude):
        layer, alt_within_layer = self._get_layer_by_altitude(altitude)
        return layer.get_temperature_by_altitude(alt_within_layer)

    def get_temperature_by_pressure(self, pressure):
        layer = self._get_layer_by_pressure(pressure)
        return layer.get_temperature_by_pressure(pressure)

    def get_pressure_by_altitude(self, altitude):
        layer, alt_within_layer = self._get_layer_by_altitude(altitude)
        return layer.get_pressure_by_altitude(alt_within_layer)

    def get_altitude_by_pressure(self, pressure):
        layer = self._get_layer_by_pressure(pressure)
        alt = layer.get_altitude_by_pressure(pressure)
        for l in self.layers:
            if l != layer:
                alt += l.height
            else:
                break
        return alt


def UniformSampler(a, b):
    def sampler():
        return a + (b - a) *  np.random.random()
    return sampler


def GaussianSampler(mean, std):
    def sampler():
        return norm.rvs(mean, std)
    return sampler


def GaussianSamplerBetween(a, b):
    mean = (a + b) / 2
    std = (b - a) / 8
    return GaussianSampler(mean, std)


def ConstantSampler(const):
    def sampler():
        return const
    return sampler


class RandomLayer:
    def __init__(self, height_sampler, lapse_rate_sampler, gas_const_sampler):
        self.height_sampler = height_sampler
        self.lapse_rate_sampler = lapse_rate_sampler
        self.gas_const_sampler = gas_const_sampler


class RandomAtmosphere:
    def __init__(self, surface_temp_sampler, surface_press_sampler, *layer_samplers):
        self.surface_temp_sampler = surface_temp_sampler
        self.surface_press_sampler = surface_press_sampler
        self.layer_samplers = layer_samplers

    def sample(self):
        atm = LayeredAtmosphere()

        for i, sampler in enumerate(self.layer_samplers):
            if i == 0:
                temp = self.surface_temp_sampler()
                press = self.surface_press_sampler()
            else:
                temp = press = None

            try:
                atm.add_layer(sampler.height_sampler(), sampler.lapse_rate_sampler(),
                              temp, press, sampler.gas_const_sampler())
            except (OverflowError, ValueError):  # retry if conditions are too extreme
                return self.sample()             # eventually explode if parameters are really TOO extreme

        return atm


def CompositionalInheritance(parent_slot='base'):
    """ kinda like the decorator parttern, without all the BS
    """
    def decorator(cls):
        def geta(inst, attr):
            parent = inst.__dict__.get(parent_slot, None)
            if parent is not None and hasattr(parent, attr):
                return getattr(parent, attr)
            else:
                raise AttributeError(attr)

        def seta(inst, attr, val):
            parent = inst.__dict__.get(parent_slot, None)

            if attr not in dir(inst) and parent is not None and attr in dir(parent):
                setattr(parent, attr, val)
            else:
                inst.__dict__[attr] = val

        cls.__getattr__ = geta
        cls.__setattr__ = seta
        return cls
    return decorator



def random_noise(samples_x=None, num_samples=None, num_support_vectors=10,
                 svr_c=2, svr_gamma=100, do_predictions=True):

    # basically generate random points and fit an RBF SVR to them, then predict some samples
    # predictions between x 0/1 (samples_x will be scaled) and y -1/1

    sv_x = np.array([0] + sorted(
        np.random.random(size=num_support_vectors)
    ) + [1]).reshape(-1, 1)
    sv_y = 2 * np.random.random(size=len(sv_x)) - 1

    svr = SVR(C=svr_c, gamma=svr_gamma).fit(sv_x, sv_y)
    if not do_predictions:
        return svr

    if samples_x is None:
        fn_x = np.arange(0, 1, 1 / num_samples)
    elif isinstance(samples_x, list):
        fn_x = np.array(samples_x)
    elif isinstance(samples_x, np.ndarray):
        fn_x = samples_x
    else:
        raise ValueError('samples_x must be a list or numpy array')

    fn_x = (fn_x - fn_x.min()) / (fn_x.max() - fn_x.min())
    fn_y = svr.predict(fn_x.reshape((-1, 1)))

    return fn_x, fn_y, svr


@CompositionalInheritance('atmosphere')
class NoisyLayeredAtmosphere:
    def __init__(self, atmosphere, gamma, c, num_sv, mag):
        self.atmosphere = atmosphere
        self.gamma = gamma
        self.c = c
        self.num_sv = num_sv
        self.mag = mag
        self.svr = random_noise(num_support_vectors=int(num_sv), svr_c=c,
                                svr_gamma=gamma, do_predictions=False)

    def get_noise_by_altitude(self, altitude):
        scaled = altitude / self.height
        return self.mag * self.svr.predict([[scaled]])[0]

    def get_temperature_by_altitude(self, altitude):
        temp = self.atmosphere.get_temperature_by_altitude(altitude)
        noise = self.get_noise_by_altitude(altitude)
        return temp + noise

    def get_temperature_by_pressure(self, pressure):
        altitude = self.get_altitude_by_pressure(pressure)
        return self.get_temperature_by_altitude(altitude)


@CompositionalInheritance('random_atmosphere')
class RandomNoisyAtmosphere:
    def __init__(self, random_atmosphere, gamma_sampler, c_sampler, num_sv_sampler, mag_sampler):
        self.random_atmosphere = random_atmosphere
        self.gamma_sampler = gamma_sampler
        self.c_sampler = c_sampler
        self.num_sv_sampler = num_sv_sampler
        self.mag_sampler = mag_sampler

    def sample(self):
        atmosphere = self.random_atmosphere.sample()
        return NoisyLayeredAtmosphere(
            atmosphere, self.gamma_sampler(), self.c_sampler(),
            self.num_sv_sampler(), self.mag_sampler()
        )


if __name__ == '__main__':
    atm = LayeredAtmosphere()
    atm.add_layer(12, -6.5, 273.15 + 19.0, 101325)
    atm.add_layer(2, 0.0)
    atm.add_layer(25, 2.5)
    atm.add_layer(9, -6.0)

    atm = LayeredAtmosphere().international_standard_atmosphere()

    alts = np.arange(0, atm.height, 1)
    temps = [atm.get_temperature_by_altitude(a) for a in alts]
    press = [atm.get_pressure_by_altitude(a) / 100 for a in alts]

    min_ps = atm.get_pressure_by_altitude(atm.height)
    pss = np.arange(min_ps / 100, 1013.25, 0.1)
    tss = [atm.get_temperature_by_pressure(p * 100) for p in pss]

    import matplotlib.pyplot as plt
    plt.title('altitude by pressure'); plt.plot(press, alts); plt.show()
    plt.title('altitude by temperature'); plt.plot(temps, alts); plt.show()
    plt.title('pressure by temperature'); plt.plot(tss, pss); plt.gca().invert_yaxis(); plt.show()
