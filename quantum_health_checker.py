# -*- coding: utf-8 -*-
"""
Quantum Health Checker

Evaluates quantum backend readiness, noise levels, and qubit availability.
Provides detailed diagnostics for quantum execution feasibility.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """Reason codes for quantum-to-classical fallback"""
    QUANTUM_UNAVAILABLE = "QUANTUM_UNAVAILABLE"
    QUANTUM_NOISE_EXCEEDED = "QUANTUM_NOISE_EXCEEDED"
    QUANTUM_RESOURCE_LIMIT = "QUANTUM_RESOURCE_LIMIT"
    QUANTUM_TIMEOUT = "QUANTUM_TIMEOUT"
    QUANTUM_ERROR = "QUANTUM_ERROR"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    BACKEND_OFFLINE = "BACKEND_OFFLINE"
    INSUFFICIENT_QUBITS = "INSUFFICIENT_QUBITS"
    CALIBRATION_FAILED = "CALIBRATION_FAILED"


@dataclass
class QuantumHealthStatus:
    """Health status of quantum backend"""
    available: bool
    backend_name: str
    num_qubits: int
    noise_level: float
    readiness_score: float
    issues: List[str]
    warnings: List[str]
    timestamp: float
    details: Dict[str, Any]


@dataclass
class FallbackEvent:
    """Record of a quantum-to-classical fallback event"""
    timestamp: float
    operation: str
    reason: FallbackReason
    reason_details: str
    quantum_health: Optional[QuantumHealthStatus]
    attempted_qubits: Optional[int]
    execution_time: float


class QuantumHealthChecker:
    """
    Comprehensive quantum backend health checker.
    
    Evaluates:
    - Backend availability
    - Noise levels
    - Qubit availability
    - Calibration status
    - Resource limits
    """
    
    def __init__(
        self,
        max_noise_threshold: float = 0.1,
        min_qubits_required: int = 2,
        timeout_seconds: float = 30.0
    ):
        """
        Initialize health checker.
        
        Args:
            max_noise_threshold: Maximum acceptable noise level (0-1)
            min_qubits_required: Minimum qubits needed
            timeout_seconds: Maximum time for health check
        """
        self.max_noise_threshold = max_noise_threshold
        self.min_qubits_required = min_qubits_required
        self.timeout_seconds = timeout_seconds
        
        self.fallback_events: List[FallbackEvent] = []
        self._last_health_check: Optional[QuantumHealthStatus] = None
    
    def quantum_health_check(
        self,
        backend_name: str = "qiskit_aer",
        required_qubits: Optional[int] = None
    ) -> QuantumHealthStatus:
        """
        Perform comprehensive quantum backend health check.
        
        Args:
            backend_name: Name of quantum backend to check
            required_qubits: Number of qubits required for operation
        
        Returns:
            QuantumHealthStatus with detailed diagnostics
        """
        start_time = time.time()
        issues = []
        warnings = []
        details = {}
        
        logger.info(f"Performing quantum health check for backend: {backend_name}")
        
        # Check 1: Backend availability
        available, backend_info = self._check_backend_availability(backend_name)
        if not available:
            issues.append(f"Backend {backend_name} not available")
            return QuantumHealthStatus(
                available=False,
                backend_name=backend_name,
                num_qubits=0,
                noise_level=1.0,
                readiness_score=0.0,
                issues=issues,
                warnings=warnings,
                timestamp=time.time(),
                details={"error": "Backend unavailable"}
            )
        
        details.update(backend_info)
        
        # Check 2: Qubit availability
        num_qubits = backend_info.get("num_qubits", 0)
        required = required_qubits or self.min_qubits_required
        
        if num_qubits < required:
            issues.append(f"Insufficient qubits: {num_qubits} < {required}")
        elif num_qubits < required * 2:
            warnings.append(f"Limited qubits: {num_qubits} available, {required} required")
        
        # Check 3: Noise level
        noise_level = self._estimate_noise_level(backend_name, backend_info)
        details["noise_level"] = noise_level
        
        if noise_level > self.max_noise_threshold:
            issues.append(f"Noise level too high: {noise_level:.4f} > {self.max_noise_threshold}")
        elif noise_level > self.max_noise_threshold * 0.7:
            warnings.append(f"Elevated noise level: {noise_level:.4f}")
        
        # Check 4: Backend calibration
        calibration_ok, calibration_info = self._check_calibration(backend_name, backend_info)
        details["calibration"] = calibration_info
        
        if not calibration_ok:
            warnings.append("Backend calibration may be outdated")
        
        # Check 5: Resource limits
        resource_ok, resource_info = self._check_resource_limits(backend_name, backend_info)
        details["resources"] = resource_info
        
        if not resource_ok:
            warnings.append("Resource limits may affect execution")
        
        # Check 6: Timeout
        elapsed = time.time() - start_time
        if elapsed > self.timeout_seconds:
            issues.append(f"Health check timeout: {elapsed:.2f}s > {self.timeout_seconds}s")
        
        # Calculate readiness score
        readiness_score = self._calculate_readiness_score(
            available=available,
            num_qubits=num_qubits,
            required_qubits=required,
            noise_level=noise_level,
            issues=issues,
            warnings=warnings
        )
        
        health_status = QuantumHealthStatus(
            available=available and len(issues) == 0,
            backend_name=backend_name,
            num_qubits=num_qubits,
            noise_level=noise_level,
            readiness_score=readiness_score,
            issues=issues,
            warnings=warnings,
            timestamp=time.time(),
            details=details
        )
        
        self._last_health_check = health_status
        
        logger.info(f"Health check complete: readiness={readiness_score:.2f}, "
                   f"issues={len(issues)}, warnings={len(warnings)}")
        
        return health_status
    
    def log_fallback(
        self,
        operation: str,
        reason: FallbackReason,
        reason_details: str,
        attempted_qubits: Optional[int] = None,
        execution_time: float = 0.0
    ):
        """
        Log a quantum-to-classical fallback event.
        
        Args:
            operation: Name of operation that fell back
            reason: Reason code for fallback
            reason_details: Detailed explanation
            attempted_qubits: Number of qubits attempted
            execution_time: Time spent before fallback
        """
        event = FallbackEvent(
            timestamp=time.time(),
            operation=operation,
            reason=reason,
            reason_details=reason_details,
            quantum_health=self._last_health_check,
            attempted_qubits=attempted_qubits,
            execution_time=execution_time
        )
        
        self.fallback_events.append(event)
        
        logger.warning(
            f"FALLBACK: {operation} -> {reason.value} | "
            f"{reason_details} | qubits={attempted_qubits}"
        )
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about fallback events.
        
        Returns:
            Dictionary with fallback metrics
        """
        if not self.fallback_events:
            return {
                "total_fallbacks": 0,
                "fallback_rate": 0.0,
                "reasons": {},
                "operations": {}
            }
        
        # Count by reason
        reason_counts = {}
        for event in self.fallback_events:
            reason = event.reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Count by operation
        operation_counts = {}
        for event in self.fallback_events:
            op = event.operation
            operation_counts[op] = operation_counts.get(op, 0) + 1
        
        # Average execution time before fallback
        avg_time = sum(e.execution_time for e in self.fallback_events) / len(self.fallback_events)
        
        return {
            "total_fallbacks": len(self.fallback_events),
            "fallback_rate": len(self.fallback_events) / max(1, len(self.fallback_events)),
            "reasons": reason_counts,
            "operations": operation_counts,
            "avg_time_before_fallback": avg_time,
            "most_common_reason": max(reason_counts.items(), key=lambda x: x[1])[0] if reason_counts else None
        }
    
    def get_fallback_events(
        self,
        operation: Optional[str] = None,
        reason: Optional[FallbackReason] = None,
        limit: Optional[int] = None
    ) -> List[FallbackEvent]:
        """
        Get fallback events with optional filtering.
        
        Args:
            operation: Filter by operation name
            reason: Filter by reason code
            limit: Maximum number of events to return
        
        Returns:
            List of matching fallback events
        """
        events = self.fallback_events
        
        if operation:
            events = [e for e in events if e.operation == operation]
        
        if reason:
            events = [e for e in events if e.reason == reason]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def clear_fallback_history(self):
        """Clear fallback event history"""
        self.fallback_events.clear()
        logger.info("Cleared fallback event history")
    
    # Private helper methods
    
    def _check_backend_availability(self, backend_name: str) -> tuple:
        """Check if quantum backend is available"""
        try:
            if backend_name == "qiskit_aer":
                from qiskit_aer import AerSimulator
                simulator = AerSimulator()
                return True, {
                    "num_qubits": 32,  # Aer simulator default
                    "backend_type": "simulator",
                    "version": "latest"
                }
            elif backend_name.startswith("ibm"):
                # Check IBM Quantum backend
                try:
                    from qiskit_ibm_runtime import QiskitRuntimeService
                    service = QiskitRuntimeService()
                    backend = service.backend(backend_name)
                    config = backend.configuration()
                    return True, {
                        "num_qubits": config.n_qubits,
                        "backend_type": "hardware",
                        "version": config.backend_version
                    }
                except Exception as e:
                    logger.warning(f"IBM backend check failed: {e}")
                    return False, {"error": str(e)}
            else:
                # Unknown backend
                return False, {"error": f"Unknown backend: {backend_name}"}
        except ImportError as e:
            logger.warning(f"Backend import failed: {e}")
            return False, {"error": "Qiskit not available"}
        except Exception as e:
            logger.error(f"Backend check error: {e}")
            return False, {"error": str(e)}
    
    def _estimate_noise_level(self, backend_name: str, backend_info: Dict) -> float:
        """Estimate noise level of backend"""
        if backend_info.get("backend_type") == "simulator":
            # Simulators have low noise
            return 0.01
        else:
            # Hardware has higher noise (estimate)
            return 0.05  # Placeholder, would query actual backend properties
    
    def _check_calibration(self, backend_name: str, backend_info: Dict) -> tuple:
        """Check backend calibration status"""
        # Simulators don't need calibration
        if backend_info.get("backend_type") == "simulator":
            return True, {"status": "N/A (simulator)"}
        
        # For hardware, would check last calibration time
        return True, {"status": "assumed_ok"}
    
    def _check_resource_limits(self, backend_name: str, backend_info: Dict) -> tuple:
        """Check resource limits"""
        # Check if we're within reasonable limits
        num_qubits = backend_info.get("num_qubits", 0)
        
        if num_qubits < 5:
            return False, {"warning": "Very limited qubits"}
        
        return True, {"status": "ok"}
    
    def _calculate_readiness_score(
        self,
        available: bool,
        num_qubits: int,
        required_qubits: int,
        noise_level: float,
        issues: List[str],
        warnings: List[str]
    ) -> float:
        """Calculate overall readiness score (0-1)"""
        if not available or issues:
            return 0.0
        
        score = 1.0
        
        # Penalize for insufficient qubits
        if num_qubits < required_qubits:
            score *= 0.0
        elif num_qubits < required_qubits * 2:
            score *= 0.7
        
        # Penalize for high noise
        noise_penalty = min(1.0, noise_level / self.max_noise_threshold)
        score *= (1.0 - noise_penalty * 0.5)
        
        # Penalize for warnings
        score *= (1.0 - len(warnings) * 0.1)
        
        return max(0.0, min(1.0, score))
