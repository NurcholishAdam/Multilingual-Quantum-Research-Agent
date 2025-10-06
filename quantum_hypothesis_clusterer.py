# -*- coding: utf-8 -*-
"""
Quantum Hypothesis Clustering using QAOA

Uses QuantumSocialPolicyOptimization with QAOA layers for clustering.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging
from qiskit.providers.aer.noise import NoiseModel
noise_model = NoiseModel.from_backend(backend)


logger = logging.getLogger(__name__)


class QuantumHypothesisClusterer:
    """
    Hypothesis clustering using QAOA (Quantum Approximate Optimization Algorithm).
    
    Input: Hypothesis embeddings + similarity matrix
    Output: Clustered hypotheses with QAOA-optimized grouping
    """
    
    def __init__(
        self,
        num_clusters: int = 5,
        qaoa_layers: int = 3,
        backend: str = "qiskit_aer",
        shots: int = 1024,
        max_noise_threshold: float = 0.15
    ):
        """
        Initialize quantum hypothesis clusterer.
        
        Args:
            num_clusters: Target number of clusters
            qaoa_layers: Number of QAOA layers (p parameter)
            backend: Quantum backend
            shots: Number of measurement shots
            max_noise_threshold: Maximum acceptable noise level (higher for QAOA)
        """
        self.num_clusters = num_clusters
        self.qaoa_layers = qaoa_layers
        self.backend = backend
        self.shots = shots
        self.max_noise_threshold = max_noise_threshold
        
        # Initialize health checker
        try:
            from quantum_health_checker import QuantumHealthChecker
            self.health_checker = QuantumHealthChecker(
                max_noise_threshold=max_noise_threshold,
                min_qubits_required=num_clusters
            )
        except ImportError:
            logger.warning("quantum_health_checker not available")
            self.health_checker = None
        
        self.quantum_available = self._check_quantum_availability()
        
        if self.quantum_available:
            self._initialize_quantum_components()
    
    def _check_quantum_availability(self) -> bool:
        """Check quantum availability"""
        try:
            import qiskit
            logger.info("Qiskit available for QAOA clustering")
            return True
        except ImportError:
            logger.warning("Qiskit not available, will use classical clustering")
            return False
    
    def _initialize_quantum_components(self):
        """Initialize QAOA components"""
        try:
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            from qiskit.algorithms.optimizers import COBYLA
            
            self.QuantumCircuit = QuantumCircuit
            self.simulator = AerSimulator()
            self.optimizer = COBYLA(maxiter=100)
            logger.info("QAOA components initialized")
        except Exception as e:
            logger.error(f"Failed to initialize QAOA: {e}")
            self.quantum_available = False
    
    def cluster(
        self,
        embeddings: np.ndarray,
        similarity_matrix: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Cluster hypotheses using QAOA.
        
        Args:
            embeddings: Hypothesis embeddings (N x D)
            similarity_matrix: Pairwise similarity matrix (N x N), computed if None
        
        Returns:
            Dictionary with cluster assignments and quality metrics
        """
        import time
        start_time = time.time()
        
        n_hypotheses = embeddings.shape[0]
        required_qubits = n_hypotheses  # One qubit per hypothesis
        
        if similarity_matrix is None:
            similarity_matrix = self._compute_similarity_matrix(embeddings)
        
        # Perform health check if available
        if self.health_checker:
            health = self.health_checker.quantum_health_check(
                backend_name=self.backend,
                required_qubits=required_qubits
            )
            
            if not health.available:
                from quantum_health_checker import FallbackReason
                reason = FallbackReason.QUANTUM_UNAVAILABLE
                if health.num_qubits < required_qubits:
                    reason = FallbackReason.INSUFFICIENT_QUBITS
                elif health.noise_level > self.max_noise_threshold:
                    reason = FallbackReason.QUANTUM_NOISE_EXCEEDED
                
                self.health_checker.log_fallback(
                    operation="hypothesis_clustering",
                    reason=reason,
                    reason_details=f"Health check failed: {', '.join(health.issues)}",
                    attempted_qubits=required_qubits,
                    execution_time=time.time() - start_time
                )
                
                logger.warning(f"Quantum not ready (score={health.readiness_score:.2f}), using classical clustering")
                return self._classical_cluster(embeddings, similarity_matrix)
        
        if not self.quantum_available:
            logger.warning("Quantum not available, using classical clustering")
            return self._classical_cluster(embeddings, similarity_matrix)
        
        try:
            result = self._qaoa_cluster(embeddings, similarity_matrix)
            
            # Add health info if available
            if self.health_checker and self.health_checker._last_health_check:
                health = self.health_checker._last_health_check
                result["quantum_health"] = {
                    "readiness_score": health.readiness_score,
                    "noise_level": health.noise_level,
                    "num_qubits": health.num_qubits
                }
            
            return result
        except Exception as e:
            if self.health_checker:
                from quantum_health_checker import FallbackReason
                self.health_checker.log_fallback(
                    operation="hypothesis_clustering",
                    reason=FallbackReason.QUANTUM_ERROR,
                    reason_details=f"Execution error: {str(e)}",
                    attempted_qubits=required_qubits,
                    execution_time=time.time() - start_time
                )
            
            logger.error(f"QAOA clustering failed: {e}, falling back to classical")
            return self._classical_cluster(embeddings, similarity_matrix)
    
    def _qaoa_cluster(
        self,
        embeddings: np.ndarray,
        similarity_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """QAOA-based clustering"""
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit import Parameter
        
        n_hypotheses = embeddings.shape[0]
        n_qubits = n_hypotheses  # One qubit per hypothesis
        
        logger.info(f"QAOA clustering: {n_hypotheses} hypotheses, {self.qaoa_layers} layers")
        
        # Create parameterized QAOA circuit
        beta = [Parameter(f'β_{i}') for i in range(self.qaoa_layers)]
        gamma = [Parameter(f'γ_{i}') for i in range(self.qaoa_layers)]
        
        qc = QuantumCircuit(n_qubits, n_qubits)
        
        # Initial state: superposition
        qc.h(range(n_qubits))
        
        # QAOA layers
        for layer in range(self.qaoa_layers):
            # Problem Hamiltonian (maximize within-cluster similarity)
            self._apply_problem_hamiltonian(qc, similarity_matrix, gamma[layer])
            
            # Mixer Hamiltonian
            self._apply_mixer_hamiltonian(qc, beta[layer])
        
        # Measurement
        qc.measure(range(n_qubits), range(n_qubits))
        
        # Optimize parameters
        best_params, best_clusters = self._optimize_qaoa_parameters(
            qc, similarity_matrix, beta, gamma
        )
        
        # Compute clustering quality
        purity = self._compute_clustering_purity(best_clusters, similarity_matrix)
        
        return {
            "method": "qaoa",
            "cluster_assignments": best_clusters,
            "num_clusters": len(set(best_clusters)),
            "purity": purity,
            "qaoa_parameters": best_params
        }
    
    def _apply_problem_hamiltonian(
        self,
        qc: Any,
        similarity_matrix: np.ndarray,
        gamma: Any
    ):
        """Apply problem Hamiltonian encoding clustering objective"""
        n_qubits = similarity_matrix.shape[0]
        
        # Apply ZZ interactions weighted by similarity
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if similarity_matrix[i, j] > 0:
                    # ZZ gate with angle proportional to similarity
                    qc.cx(i, j)
                    qc.rz(2 * gamma * similarity_matrix[i, j], j)
                    qc.cx(i, j)
    
    def _apply_mixer_hamiltonian(self, qc: Any, beta: Any):
        """Apply mixer Hamiltonian (X rotations)"""
        for qubit in range(qc.num_qubits):
            qc.rx(2 * beta, qubit)
    
    def _optimize_qaoa_parameters(
        self,
        qc: Any,
        similarity_matrix: np.ndarray,
        beta_params: List,
        gamma_params: List
    ) -> tuple:
        """Optimize QAOA parameters using classical optimizer"""
        from qiskit import transpile
        
        def objective_function(params):
            """Objective: maximize clustering quality"""
            # Bind parameters
            param_dict = {}
            for i, (b, g) in enumerate(zip(beta_params, gamma_params)):
                param_dict[b] = params[i]
                param_dict[g] = params[i + self.qaoa_layers]
            
            bound_qc = qc.assign_parameters(param_dict)
            transpiled_qc = transpile(bound_qc, self.simulator)
            
            # Execute
            job = self.simulator.run(transpiled_qc, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            # Compute expected clustering quality
            quality = self._evaluate_clustering_quality(counts, similarity_matrix)
            return -quality  # Minimize negative quality
        
        # Initial parameters
        initial_params = np.random.uniform(0, 2 * np.pi, 2 * self.qaoa_layers)
        
        # Optimize
        result = self.optimizer.minimize(objective_function, initial_params)
        best_params = result.x
        
        # Get best clustering from optimized circuit
        param_dict = {}
        for i, (b, g) in enumerate(zip(beta_params, gamma_params)):
            param_dict[b] = best_params[i]
            param_dict[g] = best_params[i + self.qaoa_layers]
        
        bound_qc = qc.assign_parameters(param_dict)
        transpiled_qc = transpile(bound_qc, self.simulator)
        job = self.simulator.run(transpiled_qc, shots=self.shots)
        counts = job.result().get_counts()
        
        # Extract best clustering
        best_bitstring = max(counts.items(), key=lambda x: x[1])[0]
        best_clusters = self._bitstring_to_clusters(best_bitstring)
        
        return best_params, best_clusters
    
    def _evaluate_clustering_quality(
        self,
        counts: Dict[str, int],
        similarity_matrix: np.ndarray
    ) -> float:
        """Evaluate clustering quality from measurement counts"""
        total_quality = 0.0
        total_counts = sum(counts.values())
        
        for bitstring, count in counts.items():
            clusters = self._bitstring_to_clusters(bitstring)
            quality = self._compute_clustering_purity(clusters, similarity_matrix)
            total_quality += quality * (count / total_counts)
        
        return total_quality
    
    def _bitstring_to_clusters(self, bitstring: str) -> List[int]:
        """Convert bitstring to cluster assignments"""
        # Simple: bit value determines cluster (0 or 1)
        # For more clusters, use groups of bits
        return [int(bit) for bit in bitstring]
    
    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix"""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(normalized, normalized.T)
        return similarity
    
    def _compute_clustering_purity(
        self,
        clusters: List[int],
        similarity_matrix: np.ndarray
    ) -> float:
        """Compute clustering purity score"""
        if len(set(clusters)) <= 1:
            return 0.0
        
        # Within-cluster similarity
        within_sim = 0.0
        count = 0
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if clusters[i] == clusters[j]:
                    within_sim += similarity_matrix[i, j]
                    count += 1
        
        return within_sim / (count + 1e-8)
    
    def _classical_cluster(
        self,
        embeddings: np.ndarray,
        similarity_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Classical k-means clustering fallback"""
        logger.info("Using classical k-means clustering")
        
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        purity = self._compute_clustering_purity(clusters.tolist(), similarity_matrix)
        
        return {
            "method": "classical_kmeans",
            "cluster_assignments": clusters.tolist(),
            "num_clusters": self.num_clusters,
            "purity": purity
        }
