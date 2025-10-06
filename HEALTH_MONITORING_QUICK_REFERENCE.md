# Quantum Health Monitoring - Quick Reference Card

## 🚀 Quick Start

```python
from quantum_health_checker import QuantumHealthChecker, FallbackReason

# Initialize
checker = QuantumHealthChecker(max_noise_threshold=0.1)

# Health check
health = checker.quantum_health_check("qiskit_aer", required_qubits=8)
print(f"Ready: {health.available}, Score: {health.readiness_score:.2f}")

# Log fallback
checker.log_fallback(
    operation="my_operation",
    reason=FallbackReason.QUANTUM_NOISE_EXCEEDED,
    reason_details="Noise too high",
    attempted_qubits=8
)

# Get statistics
stats = checker.get_fallback_statistics()
print(f"Fallbacks: {stats['total_fallbacks']}")
```

## 📋 Fallback Reason Codes

| Code | When to Use |
|------|-------------|
| `QUANTUM_UNAVAILABLE` | Qiskit not installed |
| `QUANTUM_NOISE_EXCEEDED` | Noise > threshold |
| `INSUFFICIENT_QUBITS` | Not enough qubits |
| `QUANTUM_RESOURCE_LIMIT` | Resource constraints |
| `QUANTUM_TIMEOUT` | Execution timeout |
| `QUANTUM_ERROR` | Execution failed |
| `MANUAL_OVERRIDE` | User disabled quantum |
| `BACKEND_OFFLINE` | Backend unavailable |
| `CALIBRATION_FAILED` | Calibration issues |

## 🔧 Configuration

```python
# Health Checker
checker = QuantumHealthChecker(
    max_noise_threshold=0.1,      # Max noise (0-1)
    min_qubits_required=2,         # Min qubits
    timeout_seconds=30.0           # Timeout
)

# Citation Walker
walker = QuantumCitationWalker(
    backend="qiskit_aer",
    max_noise_threshold=0.1
)

# Hypothesis Clusterer
clusterer = QuantumHypothesisClusterer(
    num_clusters=3,
    max_noise_threshold=0.15
)
```

## 📊 Health Status

```python
health = checker.quantum_health_check("qiskit_aer")

# Check fields
health.available          # bool: Overall availability
health.readiness_score    # float: 0-1 score
health.num_qubits        # int: Available qubits
health.noise_level       # float: 0-1 noise
health.issues            # List[str]: Critical issues
health.warnings          # List[str]: Warnings
```

## 📈 Fallback Statistics

```python
stats = checker.get_fallback_statistics()

# Available fields
stats['total_fallbacks']        # Total count
stats['fallback_rate']          # Rate (0-1)
stats['reasons']                # Dict[reason, count]
stats['operations']             # Dict[operation, count]
stats['most_common_reason']     # Most common reason
stats['avg_time_before_fallback']  # Average time
```

## 🔍 Query Fallback Events

```python
# All events
events = checker.get_fallback_events()

# Filter by operation
events = checker.get_fallback_events(operation="citation_traversal")

# Filter by reason
events = checker.get_fallback_events(reason=FallbackReason.QUANTUM_NOISE_EXCEEDED)

# Limit results
events = checker.get_fallback_events(limit=10)

# Event fields
event.timestamp          # When it happened
event.operation          # Operation name
event.reason            # FallbackReason enum
event.reason_details    # Detailed explanation
event.attempted_qubits  # Qubits attempted
event.execution_time    # Time before fallback
```

## 📊 Evaluation Metrics

```python
from evaluation_harness import EvaluationHarness

harness = EvaluationHarness()
metrics = harness.run_quantum_pipeline(agent, corpus, hypotheses)

# Check fallback metrics
if metrics.fallback_metrics:
    fb = metrics.fallback_metrics
    print(f"Fallbacks: {fb['total_fallbacks']}")
    print(f"Rate: {fb['fallback_rate']:.2%}")
    print(f"Reasons: {fb['reasons']}")
    print(f"Operations: {fb['operations']}")

# Generate report
report = harness.generate_fallback_report()
print(f"Total events: {report['total_events']}")
print(f"By reason: {report['events_by_reason']}")
print(f"Timeline: {report['timeline']}")
```

## 🧪 Testing

```bash
# Run demo
python demo_quantum_health_monitoring.py

# Run tests
pytest test_quantum_health_checker.py
```

## 🔧 Common Patterns

### Pattern 1: Pre-flight Check

```python
health = checker.quantum_health_check("qiskit_aer", required_qubits=10)
if health.readiness_score < 0.7:
    # Use classical directly
    result = classical_method()
else:
    # Try quantum
    result = quantum_method()
```

### Pattern 2: Automatic Fallback

```python
walker = QuantumCitationWalker(max_noise_threshold=0.1)
result = walker.traverse(...)  # Automatic health check & fallback
```

### Pattern 3: Monitor Fallbacks

```python
stats = checker.get_fallback_statistics()
if stats['fallback_rate'] > 0.5:
    logger.warning(f"High fallback rate: {stats['fallback_rate']:.2%}")
```

### Pattern 4: Analyze Patterns

```python
# Get events for specific reason
events = checker.get_fallback_events(
    reason=FallbackReason.QUANTUM_NOISE_EXCEEDED
)
for event in events:
    print(f"{event.operation}: {event.reason_details}")
```

## 📚 Documentation

- **Full Guide**: `QUANTUM_HEALTH_MONITORING_README.md`
- **Summary**: `HEALTH_MONITORING_ENHANCEMENT_SUMMARY.md`
- **Demo**: `demo_quantum_health_monitoring.py`
- **Index**: `INDEX.md`

## 🎯 Key Methods

| Method | Purpose |
|--------|---------|
| `quantum_health_check()` | Check backend health |
| `log_fallback()` | Log fallback event |
| `get_fallback_statistics()` | Get statistics |
| `get_fallback_events()` | Query events |
| `generate_fallback_report()` | Generate report |
| `clear_fallback_history()` | Clear history |

## ⚡ Performance

- Health check: ~0.01-0.05s
- Fallback logging: ~0.001s
- Minimal overhead
- Production ready

## 🔗 Integration

```python
# Automatic in quantum modules
walker = QuantumCitationWalker()  # Has health_checker
clusterer = QuantumHypothesisClusterer()  # Has health_checker
harness = EvaluationHarness()  # Has health_checker

# Access health checker
walker.health_checker.get_fallback_statistics()
```

## 💡 Tips

1. Set appropriate noise thresholds for your use case
2. Monitor fallback rates over time
3. Analyze common fallback reasons
4. Use readiness score for decisions
5. Log all fallback events
6. Generate periodic reports
7. Clear history when needed

---

**Quick Reference v1.1.0** | [Full Documentation](QUANTUM_HEALTH_MONITORING_README.md)
