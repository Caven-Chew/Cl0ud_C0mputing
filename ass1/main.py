# GenCC Simulation — Discrete Event Simulator
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Network parameters
C = 100e6  # Link capacity (bps)
MSS = 1500  # Max segment size (bytes)
K = 50  # ECN threshold (packets)
QMAX = 200  # Queue limit (packets)
G = 0.0625  # DCTCP smoothing factor


def simulate_protocol(protocol, rtt_ms, n_flows=5, duration=10.0):
    """Simplified fluid model simulation."""
    rtt = rtt_ms / 1000.0
    cwnd = np.ones(n_flows) * 10  # initial window (MSS)
    alpha = np.zeros(n_flows)  # DCTCP ECN fraction
    beta = np.ones(n_flows) * 0.5  # GenCC adaptive decrease factor
    w = np.array([0.3, -0.4, 0.1])  # GenCC feature weights
    throughputs = []
    queue = 0.0
    dt = rtt / 10.0
    t = 0.0
    while t < duration:
        # Aggregate sending rate
        total_rate = np.sum(cwnd) * MSS * 8 / rtt
        total_rate = min(total_rate, C)
        # Queue dynamics (fluid model)
        queue += (total_rate / (MSS * 8) - C / (MSS * 8)) * dt
        queue = max(0, min(queue, QMAX))
        ecn_fraction = min(1.0, max(0.0, (queue - K) / K)) if queue > K else 0.0
        for i in range(n_flows):
            if protocol == "cubic":
                if ecn_fraction > 0.1 or queue > QMAX * 0.9:
                    cwnd[i] = max(1, cwnd[i] * 0.5)
                else:
                    cwnd[i] += 1.0 / cwnd[i]
            elif protocol == "dctcp":
                alpha[i] = (1 - G) * alpha[i] + G * ecn_fraction
                if ecn_fraction > 0:
                    cwnd[i] = max(1, cwnd[i] * (1 - alpha[i] / 2))
                else:
                    cwnd[i] += 1.0 / cwnd[i]
            elif protocol == "gencc":
                # Adaptive beta using learned weights
                features = np.array([rtt_ms / 600, ecn_fraction, queue / QMAX])
                raw = np.dot(w, features)
                beta[i] = 1 / (1 + np.exp(-raw))  # sigmoid -> (0,1)
                beta[i] = np.clip(beta[i], 0.3, 0.8)
                if ecn_fraction > 0:
                    cwnd[i] = max(1, cwnd[i] * (1 - beta[i] * ecn_fraction))
                else:
                    cwnd[i] += 1.0 / cwnd[i]
        throughputs.append(total_rate / 1e6)  # Mbps
        t += dt
    return np.mean(throughputs)


rtts = [20, 50, 100, 200, 400, 600]
results = {p: [] for p in ["cubic", "dctcp", "gencc"]}
for rtt in rtts:
    for p in results:
        results[p].append(simulate_protocol(p, rtt))

plt.figure(figsize=(8, 5))
plt.plot(rtts, results["cubic"], "b-o", label="TCP CUBIC")
plt.plot(rtts, results["dctcp"], "r-s", label="DCTCP")
plt.plot(rtts, results["gencc"], "g-^", label="GenCC (proposed)")
plt.xlabel("RTT (ms)", fontsize=12)
plt.ylabel("Throughput (Mbps)", fontsize=12)
plt.title("Throughput vs. RTT: 5 Flows, 100 Mbps Link", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("throughput_vs_rtt.png", dpi=150)
print("Results saved.")
for rtt, c, d, g in zip(rtts, results["cubic"], results["dctcp"], results["gencc"]):
    print(f"RTT={rtt}ms  CUBIC={c:.1f}  DCTCP={d:.1f}  GenCC={g:.1f}")
