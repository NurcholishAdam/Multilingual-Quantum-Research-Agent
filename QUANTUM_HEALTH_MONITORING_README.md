# Quantum Health Monitoring & Fallback Tracking

Comprehensive quantum backend health checking and fallback event tracking for the Multilingual Quantum Research Agent.

## 🏥 Features

### Quantum Health Checking
- **Backend Readiness**: Evaluate quantum backend availability
- **Noise Level Assessment**: Measure and threshold noise levels
- **Qubit Availability**: Check sufficient qubits for operations
- **Calibration Status**: Verify backend calibration
- **Resource Limits**: Detect resource constraints
- **Readiness Score**: Overall health score (0-1)

### Fallback Logging
- **Reason Codes**: Structured fallback reasons
- **Event Tracking**: Complete fallback event history
- **Detailed Diagnostics**: Health status at fallback time
- **Performance Metrics**: Time spent before fallback

### Benchmarking Integration
- **Fallback Metrics**: Track fallback frequency and reasons
- **Comparative Analysis**: Quantum vs. classical with fallback data
- **Trend Analysis**: Identify fallback patterns
- **Reporting**: Comprehensive fallback reports

## 📋 Fallback Reason Codes

| Code | Description | Trigger |
|------|-------------|---------|
| `QUANTUM_UNAVAILABLE` | Quantum backend not available | Qiskit not installed or backend offline |
| `QUANTUM_NOISE_EXCEEDED` | Noise level too high | Noise > threshold |
| `QUANTUM_RESOURCE_LIMIT` | Resource limits exceeded | Queue full, memory limit |
| `QUANTUM_TIMEOUT` | Operation timeout | Execution time > limit |
| `QUANTUM_ERROR` | Execution error | Circuit compilation/execution failed |
| `MANUAL_OVERRIDE` | User disabled quantum | `quantum_enabled=False` |
| `BACKEND_OFFLINE` | Backend unavailable | Hardware maintenance |
| `INSUFFICIENT_QUBITS` | Not enough qubits | Required > available |
| `CALIBRATION_FAILED` | Calibration issues | Backend not calibrated |

## 🚀 Quick Start

### Basic Health Check

```python
from quantum_health_checker import QuantumHealthChecker

# Initialize checker
checker = QuantumHealthChecker(
    max_noise_threshold=0.1,
    min_qubits_required=4
)

# Perform health check
health = checker.quantum_health_check(
    backend_name="qiskit_aer",
    required_qubits=8
)

print(f"Available: {health.available}")
print(f"Readiness Score: {health.readiness_score:.2f}")
print(f"Noise Level: {health.noise_level:.4f}")
print(f"Qubits: {health.num_qubits}")

if health.issues:
    print(f"Issues: {', '.join(health.issues)}")
```

### Fallback Logging

```python
from quantum_health_checker import FallbackReason

# Log a fallback event
checker.log_fallback(
    operation="citation_traversal",
    reason=FallbackReason.QUANTUM_NOISE_EXCEEDED,
    reason_details="Noise level 0.15 exceeds threshold 0.10",
    attempted_qubits=10,
    execution_time=2.5
)

# Get statistics
stats = checker.get_fallback_statistics()
print(f"Total fallbacks: {stats['total_fallbacks']}")
print(f"Most common reason: {stats['most_common_reason']}")
```

### Integrated Usage

```python
from quantum_citation_walker import QuantumCitationWalker

# Walker automatically uses health checker
walker = QuantumCitationWalker(
    backend="qiskit_aer",
    max_noise_threshold=0.1
)

# Health check performed automatically
result = walker.traverse(
    adjacency_matrix=adj_matrix,
    semantic_weights=weights,
    start_nodes=[0],
    max_steps=5
)

# Check if quantum was used
print(f"Method: {result['method']}")

# Get health info
if "quantum_health" in result:
    health = result["quantum_health"]
    print(f"Readiness: {health['readiness_score']:.2f}")
```

## 📊 Health Status Fields

### QuantumHealthStatus

```python
@dataclass
class QuantumHealthStatus:
    available: bool              # Overall availability
    backend_name: str            # Backend identifier
    num_qubits: int             # Available qubits
    noise_level: float          # Estimated noise (0-1)
    readiness_score: float      # Overall score (0-1)
    issues: List[str]           # Critical issues
    warnings: List[str]         # Non-critical warnings
    timestamp: float            # Check timestamp
    details: Dict[str, Any]     # Additional info
```

### FallbackEvent

```python
@dataclass
class FallbackEvent:
    timestamp: float                        # Event time
    operation: str                          # Operation name
    reason: FallbackReason                  # Reason code
    reason_details: str                     # Detailed explanation
    quantum_health: QuantumHealthStatus     # Health at fallback
    attempted_qubits: int                   # Qubits attempted
    execution_time: float                   # Time before fallback
```

## 🔧 Configuration

### Health Checker Parameters

```python
checker = QuantumHealthChecker(
    max_noise_threshold=0.1,      # Maximum acceptable noise
    min_qubits_required=2,         # Minimum qubits needed
    timeout_seconds=30.0           # Health check timeout
)
```

### Quantum Module Integration

```python
# Citation Walker
walker = QuantumCitationWalker(
    backend="qiskit_aer",
    shots=1024,
    max_noise_threshold=0.1        # Noise threshold
)

# Hypothesis Clusterer
clusterer = QuantumHypothesisClusterer(
    num_clusters=3,
    qaoa_layers=2,
    max_noise_threshold=0.15       # Higher threshold for QAOA
)
```

## 📈 Evaluation Metrics

### Fallback Metrics in Evaluation

```python
from evaluation_harness import EvaluationHarness

harness = EvaluationHarness()

# Run evaluation
quantum_metrics = harness.run_quantum_pipeline(agent, corpus, hypotheses)

# Check fallback metrics
if quantum_metrics.fallback_metrics:
    fb = quantum_metrics.fallback_metrics
    print(f"Total fallbacks: {fb['total_fallbacks']}")
    print(f"Fallback rate: {fb['fallback_rate']:.2%}")
    print(f"Reasons: {fb['reasons']}")
```

### Fallback Report

```python
# Generate comprehensive report
report = harness.generate_fallback_report()

print(f"Total events: {report['total_events']}")
print(f"Events by reason: {report['events_by_reason']}")
print(f"Events by operation: {report['events_by_operation']}")
print(f"Timeline: {report['timeline']}")
```

## 📊 Example Output

### Health Check Output

```
Performing quantum health check for backend: qiskit_aer
Health check complete: readiness=0.95, issues=0, warnings=0

Available: True
Readiness Score: 0.95
Noise Level: 0.0100
Qubits: 32
```

### Fallback Statistics

```
Fallback Statistics:
  Total fallbacks: 5
  Most common reason: QUANTUM_NOISE_EXCEEDED
  Avg time before fallback: 2.34s

Fallbacks by reason:
  QUANTUM_NOISE_EXCEEDED: 2
  INSUFFICIENT_QUBITS: 2
  QUANTUM_TIMEOUT: 1

Fallbacks by operation:
  citation_traversal: 3
  hypothesis_clustering: 2
```

### Evaluation with Fallbacks

```
Quantum pipeline completed in 2.30s
Fallbacks: 2 (rate=20.00%)

Fallback Metrics:
  Total fallbacks: 2
  Fallback rate: 20.00%
  Most common reason: QUANTUM_NOISE_EXCEEDED
  Reasons breakdown:
    QUANTUM_NOISE_EXCEEDED: 1
    INSUFFICIENT_QUBITS: 1
```

## 🎯 Use Cases

### 1. Production Monitoring

```python
# Monitor quantum backend health in production
checker = QuantumHealthChecker()

# Periodic health checks
health = checker.quantum_health_check("qiskit_aer")

if health.readiness_score < 0.7:
    # Alert or switch to classical
    logger.warning(f"Low readiness: {health.readiness_score:.2f}")
```

### 2. Noise Threshold Tuning

```python
# Test different noise thresholds
thresholds = [0.05, 0.1, 0.15, 0.2]

for threshold in thresholds:
    checker = QuantumHealthChecker(max_noise_threshold=threshold)
    health = checker.quantum_health_check("qiskit_aer")
    print(f"Threshold {threshold}: readiness={health.readiness_score:.2f}")
```

### 3. Fallback Analysis

```python
# Analyze fallback patterns
stats = checker.get_fallback_statistics()

if stats['total_fallbacks'] > 10:
    # Investigate most common reason
    reason = stats['most_common_reason']
    events = checker.get_fallback_events(reason=FallbackReason[reason])
    
    for event in events:
        print(f"{event.operation}: {event.reason_details}")
```

### 4. Resource Planning

```python
# Check qubit requirements
operations = [
    ("small_graph", 8),
    ("medium_graph", 16),
    ("large_graph", 32)
]

for name, qubits in operations:
    health = checker.quantum_health_check(
        backend_name="qiskit_aer",
        required_qubits=qubits
    )
    print(f"{name} ({qubits} qubits): {health.available}")
```

## 🔍 Debugging

### Enable Detailed Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("quantum_health_checker")
logger.setLevel(logging.DEBUG)
```

### Inspect Fallback Events

```python
# Get recent fallback events
events = checker.get_fallback_events(limit=10)

for event in events:
    print(f"Time: {event.timestamp}")
    print(f"Operation: {event.operation}")
    print(f"Reason: {event.reason.value}")
    print(f"Details: {event.reason_details}")
    if event.quantum_health:
        print(f"Health score: {event.quantum_health.readiness_score:.2f}")
    print()
```

### Export Fallback Data

```python
import json

# Export fallback statistics
stats = checker.get_fallback_statistics()
with open("fallback_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# Export fallback events
events = checker.get_fallback_events()
events_data = [
    {
        "timestamp": e.timestamp,
        "operation": e.operation,
        "reason": e.reason.value,
        "details": e.reason_details,
        "qubits": e.attempted_qubits
    }
    for e in events
]
with open("fallback_events.json", "w") as f:
    json.dump(events_data, f, indent=2)
```

## 🧪 Testing

### Run Health Monitoring Demo

```bash
python demo_quantum_health_monitoring.py
```

This runs 6 comprehensive demos:
1. Quantum health check
2. Fallback logging
3. Citation walker with health monitoring
4. Evaluation with fallback metrics
5. Comprehensive fallback report
6. Noise threshold testing

### Unit Tests

```python
# Test health checker
def test_health_check():
    checker = QuantumHealthChecker()
    health = checker.quantum_health_check("qiskit_aer")
    assert health.available
    assert 0 <= health.readiness_score <= 1

# Test fallback logging
def test_fallback_logging():
    checker = QuantumHealthChecker()
    checker.log_fallback(
        operation="test",
        reason=FallbackReason.QUANTUM_ERROR,
        reason_details="Test error"
    )
    stats = checker.get_fallback_statistics()
    assert stats['total_fallbacks'] == 1
```

## 📚 API Reference

### QuantumHealthChecker

#### Methods

- `quantum_health_check(backend_name, required_qubits)`: Perform health check
- `log_fallback(operation, reason, reason_details, ...)`: Log fallback event
- `get_fallback_statistics()`: Get fallback statistics
- `get_fallback_events(operation, reason, limit)`: Get fallback events
- `clear_fallback_history()`: Clear event history

### EvaluationHarness

#### New Methods

- `generate_fallback_report()`: Generate comprehensive fallback report
- `_generate_fallback_summary(quantum_results)`: Summarize fallbacks

#### Updated Methods

- `run_quantum_pipeline()`: Now includes fallback metrics
- `_generate_summary()`: Includes fallback summary

## 🔗 Integration

### With Existing Components

- **quantum_citation_walker.py**: Automatic health checking
- **quantum_hypothesis_clusterer.py**: Health-aware clustering
- **evaluation_harness.py**: Fallback metrics tracking
- **multilingual_research_agent.py**: Agent-level health monitoring

### With External Systems

- **Monitoring Systems**: Export metrics to Prometheus, Grafana
- **Alerting**: Trigger alerts on high fallback rates
- **Logging**: Integrate with centralized logging (ELK, Splunk)

## 📊 Best Practices

1. **Set Appropriate Thresholds**: Tune noise and qubit thresholds for your use case
2. **Monitor Fallback Rates**: Track fallback frequency over time
3. **Analyze Patterns**: Identify common fallback reasons
4. **Optimize Resources**: Adjust operations based on health checks
5. **Log Everything**: Maintain comprehensive fallback logs
6. **Regular Health Checks**: Perform periodic backend health checks
7. **Graceful Degradation**: Always have classical fallback ready

## 🎓 Advanced Topics

### Custom Health Checks

```python
class CustomHealthChecker(QuantumHealthChecker):
    def _estimate_noise_level(self, backend_name, backend_info):
        # Custom noise estimation
        if backend_name == "my_backend":
            return self._query_backend_noise()
        return super()._estimate_noise_level(backend_name, backend_info)
```

### Predictive Fallback

```python
# Predict if operation will fallback
health = checker.quantum_health_check("qiskit_aer", required_qubits=20)

if health.readiness_score < 0.5:
    # Skip quantum, use classical directly
    result = classical_method()
else:
    # Try quantum
    result = quantum_method()
```

---

**Status**: ✅ Complete

**Version**: 1.1.0

**Last Updated**: 2025-10-06
