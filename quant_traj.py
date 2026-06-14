import numpy as np
import qutip as qt

# Quantum simulation: Multi-mode bosonic system with Lindblad for loss/measurement/feedback
# Simplified for demo: truncated Fock basis per mode, photon-count dependent damping + feedback

def create_multi_mode_hamiltonian(n_modes=2, trunc=5):
    """Simple effective Hamiltonian from cost function (mean-field like)"""
    a_list = [qt.tensor([qt.destroy(trunc) if j==i else qt.qeye(trunc) for j in range(n_modes)]) for i in range(n_modes)]
    H = 0
    for i in range(n_modes):
        n_op = a_list[i].dag() * a_list[i]
        H += 0.1 * n_op  # placeholder quadratic terms
    return H, a_list

def photon_dependent_loss_ops(a_list, x_current, gamma=0.1):
    """Engineered loss: rate depends on photon counts (approximating measurement-feedback)"""
    ops = []
    for i, a in enumerate(a_list):
        rate = gamma * (1 + 2 * x_current[i])  # multiplicative-like
        ops.append(np.sqrt(rate) * a)
    return ops

def simulate_quantum_trajectory(n_modes=2, trunc=4, n_steps=50, n_traj=10, eta=0.02, seed=42):
    np.random.seed(seed)
    H, a_list = create_multi_mode_hamiltonian(n_modes, trunc)
    
    # Initial coherent-like state on normalized photon budget
    psi0 = qt.tensor([qt.coherent(trunc, 1.0) for _ in range(n_modes)])
    psi0 = psi0.unit()
    
    trajectories = []
    for traj in range(n_traj):
        x_traj = []
        rho = psi0 * psi0.dag()
        x = np.array([qt.expect(a.dag()*a, rho) for a in a_list])
        x /= np.sum(x) + 1e-12  # normalize to simplex
        x_traj.append(x.copy())
        
        for step in range(n_steps):
            c_ops = photon_dependent_loss_ops(a_list, x)
            
            # Solve master eq for one step
            result = qt.mesolve(H, rho, [0, eta], c_ops=c_ops)
            rho = result.states[-1]
            
            # Update x from expectations
            x = np.array([qt.expect(a.dag()*a, rho) for a in a_list])
            total = np.sum(x)
            if total > 0:
                x /= total
            else:
                x = np.ones(n_modes) / n_modes
            x_traj.append(x.copy())
        trajectories.append(np.array(x_traj))
    return trajectories

if __name__ == "__main__":
    trajs = simulate_quantum_trajectory(n_modes=3, n_steps=30, n_traj=5)
    print("Quantum simulation complete. Trajectories shape:", np.array(trajs).shape)
    print("QuTiP version:", qt.__version__)
