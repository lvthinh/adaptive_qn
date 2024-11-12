import jax
import jax.numpy as jnp
from queso.sensors import Sensor
from queso.estimators import BayesianDNNEstimator

sensor = Sensor(n=4, k=4)

theta, phi, mu = sensor.theta, sensor.phi, sensor.mu
sensor.qfi(theta, phi)
sensor.cfi(theta, phi, mu)
sensor.state(theta, phi, mu)

data = sensor.sample(theta, phi, mu, n_shots=10)

estimator = BayesianDNNEstimator()
posterior = estimator(data)