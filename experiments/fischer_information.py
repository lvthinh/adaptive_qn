import jax
import uuid
from functools import partial

from qsense.functions import *
from qsense.functions import initialize, compile

if __name__ == "__main__":
    n = 3  # number of particles
    d = 2  # local dimensions

    # layer = [(z, "phase1"), (x, "phase2")]
    # layer = [(phase, uuid.uuid4()) for _ in range(n)]

    ket_i = nketx0(n)
    # ket_i = nket_ghz(n)

    circuit = []
    circuit.append([(u2, str(uuid.uuid4())) for i in range(n)])
    circuit.append([(phase, "phase") for i in range(n)])
    # circuit.append([(rx, str(uuid.uuid4())) for i in range(n)])
    # layer2 = [(rx, "phase") for i in range(n)]
    # circuit = [layer1, layer2]

    params = initialize(circuit)
    params["phase"] = np.array([(0.25 / 4) * np.pi + 0j])

    compile = jax.jit(partial(compile, circuit=circuit))
    u = compile(params)

    for i in range(10):
        params["phase"] = np.array([(i / 4) * np.pi + 0j])

        u = compile(params)
        ket_f = u @ ket_i

        # print(circuit)
        print(params)
        print(ket_f)

    # keys = ["phase"]
    # # keys = params.keys()
    # # qf = qfim(params, circuit, ket_i, ["phase"])
    # # print(qf)
    #
    # f = qfim(params, circuit, ket_i, keys)
    # print(f)
    # # sns.heatmap(f)
    # # plt.show()
