import torch

def von_neumann_entropy(rho, eps=1e-10):
    """Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ)."""
    eigenvalues = torch.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > eps]  # Clip near-zero for stability
    return -torch.sum(eigenvalues * torch.log(eigenvalues)).real

def relative_coherence(rho, eps=1e-10):
    """Compute relative entropy of coherence C_rel(ρ) = S(ρ_diag) - S(ρ)."""
    diag_elements = torch.diag(rho).real.clamp(min=eps)
    diag_elements /= diag_elements.sum()  # Normalize to probabilities
    diag_rho = torch.diag(diag_elements)
    diag_entropy = von_neumann_entropy(diag_rho, eps)
    full_entropy = von_neumann_entropy(rho, eps)
    return diag_entropy - full_entropy

def project_psd_unit_trace(mat, eps=1e-10):
    """Project matrix to PSD with unit trace (simple eigendecomp method)."""
    eigenvalues, eigenvectors = torch.linalg.eigh(mat)
    eigenvalues = eigenvalues.real.clamp(min=0)  # Clip negative eigenvalues
    eigenvalues /= (eigenvalues.sum() + eps)  # Normalize trace to 1
    projected = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T.conj()
    return projected.real  # Assume real for language model context

def free_energy_descent_step(rho, E_func, T=1.0, eta=0.01, eps=1e-10):
    """Single step of coherence-aware free-energy descent:
    ρ_{t+1} = Π [ ρ_t - η ∇_ρ (E(ρ_t) - T C_rel(ρ_t)) ]
    """
    rho.requires_grad_(True)
    E = E_func(rho)
    C_rel = relative_coherence(rho, eps)
    F = E - T * C_rel  # Free energy F_T(ρ)
    F.backward()
    with torch.no_grad():
        rho_update = rho - eta * rho.grad
        rho_update = project_psd_unit_trace(rho_update, eps)
    rho.requires_grad_(False)
    return rho_update

# Example usage: Small 2x2 density matrix and dummy task energy
dim = 2
rho_init = torch.eye(dim) / dim  # Initial maximally mixed state (high entropy)

def dummy_E(rho):
    """Dummy task energy surrogate: -log Tr(ρ @ target), simulating eq. (8)."""
    # Arbitrary target operator (e.g., a POVM element)
    target = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    return -torch.log(torch.trace(rho @ target) + 1e-10)

# Perform one update step
rho_updated = free_energy_descent_step(rho_init, dummy_E, T=1.0, eta=0.1)

# Print results for verification
print("Initial ρ:\n", rho_init)
print("\nUpdated ρ:\n", rho_updated)
print("\nInitial von Neumann Entropy:", von_neumann_entropy(rho_init))
print("Updated von Neumann Entropy:", von_neumann_entropy(rho_updated))
print("Initial Relative Coherence:", relative_coherence(rho_init))
print("Updated Relative Coherence:", relative_coherence(rho_updated))