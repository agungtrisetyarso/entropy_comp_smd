import numpy as np
import matplotlib.pyplot as plt

def quadratic_cost(x, c, J):
    return np.dot(c, x) + np.dot(x, np.dot(J, x))

def grad_E(x, c, J):
    return c + 2 * np.dot(J, x)

def simulate_smd(n=20, n_steps=300, n_traj=30, eta=0.03, sigma=0.08, momentum=0.0, seed=42):
    """Standard or momentum-accelerated SMD"""
    np.random.seed(seed)
    c = np.random.randn(n) * 0.1
    J = np.random.randn(n, n) * 0.08  # stronger nonconvexity
    J = (J + J.T) / 2
    
    costs = []
    for traj in range(n_traj):
        x = np.ones(n) / n
        v = np.zeros(n)  # velocity for momentum
        traj_costs = []
        for t in range(n_steps):
            g = grad_E(x, c, J)
            # Momentum / Nesterov-like for mirror descent
            if momentum > 0:
                v = momentum * v - eta * g
                log_update = v + sigma * np.random.randn(n)
            else:
                log_update = -eta * g + sigma * np.random.randn(n)
            x_new = x * np.exp(log_update)
            x_new /= np.sum(x_new) + 1e-12
            x = np.maximum(x_new, 1e-12)
            x /= np.sum(x)
            traj_costs.append(quadratic_cost(x, c, J))
        costs.append(traj_costs)
    return np.array(costs), c, J

def compare_convergence():
    n_steps = 300
    # Standard SMD
    costs_std, _, _ = simulate_smd(n_steps=n_steps, momentum=0.0, seed=42)
    # Accelerated / Momentum SMD
    costs_acc, _, _ = simulate_smd(n_steps=n_steps, momentum=0.9, seed=42)
    
    mean_std = np.mean(costs_std, axis=0)
    mean_acc = np.mean(costs_acc, axis=0)
    
    # Find iterations to reach 80% of final improvement
    final_std = mean_std[-1]
    final_acc = mean_acc[-1]
    init = mean_std[0]
    
    def iters_to_target(costs, target):
        for i, c in enumerate(costs):
            if c <= target:
                return i
        return len(costs)
    
    target = init * 0.2 + final_std * 0.8  # 80% progress
    iters_std = iters_to_target(mean_std, target)
    iters_acc = iters_to_target(mean_acc, target)
    
    speedup = (iters_std - iters_acc) / iters_std * 100 if iters_std > 0 else 0
    
    print("=== Accelerated SMD Prediction Test ===")
    print(f"Standard SMD final cost: {final_std:.4f}")
    print(f"Accelerated SMD final cost: {final_acc:.4f}")
    print(f"Iterations to target (std): {iters_std}")
    print(f"Iterations to target (acc): {iters_acc}")
    print(f"Speedup: {speedup:.1f}% faster convergence")
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(mean_std, label='Standard SMD (device-like)', linewidth=2)
    plt.plot(mean_acc, label='Momentum-Accelerated SMD (Prediction)', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cost E(x)')
    plt.title('Prediction: Accelerated Mirror Descent Converges Faster')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/workdir/artifacts/accelerated_smd_prediction.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Plot saved: accelerated_smd_prediction.png")

if __name__ == "__main__":
    compare_convergence()
