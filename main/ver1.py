import jax
import jax.numpy as jnp
import optax
import h5py
import tqdm
import matplotlib.pyplot as plt
from tensorcircuit import Circuit, DMCircuit
from tensorcircuit.noisemodel import depolarizing, amplitude_damping
from queso.sensors.tc.sensor import Sensor
from queso.configs import Configuration
from queso.io import IO

# Function to apply a noisy channel (depolarizing and amplitude damping)
def apply_channel(circuit, depolarize_prob=0.05, loss_prob=0.05):
    """
    Apply depolarizing and amplitude damping noise to the circuit.
    
    Args:
        circuit (DMCircuit): Circuit instance.
        depolarize_prob (float): Probability of depolarizing noise.
        loss_prob (float): Probability of amplitude damping.
    """
    depolarizing_noise = depolarizing(depolarize_prob)
    amplitude_damping_noise = amplitude_damping(loss_prob)
    circuit.apply_general_kraus(0, depolarizing_noise)
    circuit.apply_general_kraus(0, amplitude_damping_noise)
    return circuit

# Define the encoding circuit using the parameter theta
def encoding_circuit(theta):
    """
    Creates a variational encoding circuit with parameter theta.

    Args:
        theta (jnp.array): Parameter for encoding.

    Returns:
        Circuit: Quantum circuit encoding the message with theta.
    """
    circuit = Circuit(1)
    circuit.ry(0, theta)  # Variational encoding using Ry
    return circuit

# Define the POVM measurement circuit using the parameter mu
def povm_measurement_circuit(mu):
    """
    Creates a POVM measurement circuit using parameter mu.

    Args:
        mu (jnp.array): Parameter for POVM measurement.

    Returns:
        Circuit: Quantum circuit for measurement.
    """
    circuit = Circuit(1)
    circuit.rx(0, mu)  # Example of variational POVM using Rx
    return circuit

# Define a simple QEC scheme
def qec_encode(circuit):
    """
    Apply a simple QEC encoding (e.g., repetition code).

    Args:
        circuit (Circuit): Quantum circuit instance.
    """
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    return circuit

def qec_decode(circuit):
    """
    Apply QEC decoding.

    Args:
        circuit (Circuit): Quantum circuit instance.
    """
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    return circuit

# Perform one communication round from sender through repeater to receiver
def communication_round(sensor, theta, mu, phi, optimizer, opt_state, loss_fn, depolarize_prob, loss_prob):
    """
    Perform one round of communication with noisy channel, QEC, and parameter updates.
    
    Args:
        sensor (Sensor): Sensor instance.
        theta (jnp.array): Encoding parameter.
        mu (jnp.array): Measurement parameter.
        phi (jnp.array): Channel parameter for QFI/CFI.
        optimizer (optax.GradientTransformation): Optimizer instance.
        opt_state (optax.OptState): Optimizer state.
        loss_fn (callable): Loss function.
        depolarize_prob (float): Depolarizing probability.
        loss_prob (float): Amplitude damping probability.

    Returns:
        tuple: Updated theta, mu, opt_state, and loss value.
    """
    params = {"theta": theta, "mu": mu}

    # Step 1: Encode the message at the sender using theta
    circuit = DMCircuit(3)
    encoding = encoding_circuit(theta)
    circuit.append(encoding)

    # Step 2: Apply QEC encoding
    qec_encode(circuit)

    # Step 3: Apply the noisy channel
    apply_channel(circuit, depolarize_prob, loss_prob)

    # Step 4: At the repeater, apply POVM measurement using mu
    povm_circuit = povm_measurement_circuit(mu)
    circuit.append(povm_circuit)

    # Step 5: Update theta and mu based on the Fisher Information using gradient descent
    val, grads = jax.value_and_grad(loss_fn)(params, phi)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    # Step 6: Re-encode and send to the receiver
    encoding = encoding_circuit(params["theta"])
    circuit.append(encoding)
    qec_encode(circuit)  # Re-encode with QEC before sending to receiver

    return params["theta"], params["mu"], opt_state, val

# Simulate the full network across multiple rounds
def simulate_network(io, config, key, n_rounds=100, depolarize_prob=0.05, loss_prob=0.05):
    # Initialization
    n = config.n
    k = config.k
    phi_range = config.phi_range
    phi = jnp.array(config.phi_fi)
    theta = jax.random.uniform(key, shape=(n,))
    mu = jax.random.uniform(key, shape=(k,))
    lr = config.lr_circ
    optimizer = optax.adam(learning_rate=lr)
    
    # Initialize sensor and optimizer
    sensor = Sensor(n, k, **config.sensor_params)
    opt_state = optimizer.init({"theta": theta, "mu": mu})

    # Define loss functions based on CFI/QFI
    if config.loss_fi == "loss_cfi":
        loss_fn = lambda params, phi: -sensor.cfi(params["theta"], phi, params["mu"])
    elif config.loss_fi == "loss_qfi":
        loss_fn = lambda params, phi: -sensor.qfi(params["theta"], phi)
    else:
        raise ValueError("Invalid loss function. Choose either 'loss_cfi' or 'loss_qfi'.")

    # Logging metrics
    theta_vals, mu_vals, losses = [], [], []

    # Communication rounds
    for i in tqdm.tqdm(range(n_rounds), desc="Communication Rounds"):
        # Perform one communication round
        theta, mu, opt_state, loss = communication_round(
            sensor, theta, mu, phi, optimizer, opt_state, loss_fn, depolarize_prob, loss_prob
        )
        
        # Store metrics
        theta_vals.append(theta)
        mu_vals.append(mu)
        losses.append(-loss)
        
        # Print the progress
        if i % 10 == 0:
            print(f"Round {i}: Loss = {-loss:.4f}")

    # Save results
    with h5py.File(io.path / "simulation_results.h5", "w") as hf:
        hf.create_dataset("theta_vals", data=jnp.array(theta_vals))
        hf.create_dataset("mu_vals", data=jnp.array(mu_vals))
        hf.create_dataset("losses", data=jnp.array(losses))

    # Plot results
    plt.plot(losses, label="Loss")
    plt.xlabel("Communication Round")
    plt.ylabel("Fisher Information")
    plt.legend()
    plt.show()
    
    return theta_vals, mu_vals, losses

# Main function to set up IO and configuration and run the simulation
if __name__ == "__main__":
    # Set up IO and configuration
    io = IO(folder="simulation_output")
    config = Configuration.from_yaml(io.path / "config.yaml")
    key = jax.random.PRNGKey(config.seed)
    
    # Run the simulation
    n_rounds = 100  # Number of communication rounds
    theta_vals, mu_vals, losses = simulate_network(io, config, key, n_rounds=n_rounds, depolarize_prob=0.05, loss_prob=0.05)
    
    print("Simulation complete. Results saved.")
