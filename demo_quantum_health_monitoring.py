# -*- coding: utf-8 -*-
"""
Demo: Quantum Health Monitoring and Fallback Tracking

Demonstrates:
1. Quantum health checking
2. Fallback logging with reason codes
3. Fallback metrics in benchmarking
4. Comprehensive fallback reporting
"""

import logging
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from quantum_health_checker import QuantumHealthChecker, FallbackReason
from quantum_citation_walker import QuantumCitationWalker
from quantum_hypothesis_clusterer import QuantumHypothesisClusterer
from synthetic_data_generator import SyntheticDataGenerator
from evaluation_harness import EvaluationHarness
from multilingual_research_agent import MultilingualResearchAgent, Language, Hypothesis


def demo_health_check():
    """Demonstrate quantum health checking"""
    logger.info("=" * 80)
    logger.info("DEMO 1: Quantum Health Check")
    logger.info("=" * 80)
    
    checker = QuantumHealthChecker(
        max_noise_threshold=0.1,
        min_qubits_required=4
    )
    
    # Check different backends
    backends = ["qiskit_aer", "ibmq_qasm_simulator", "unknown_backend"]
    
    for backend in backends:
        logger.info(f"\nChecking backend: {backend}")
        health = checker.quantum_health_check(
            backend_name=backend,
            required_qubits=8
        )
        
        logger.info(f"  Available: {health.available}")
        logger.info(f"  Readiness Score: {health.readiness_score:.2f}")
        logger.info(f"  Qubits: {health.num_qubits}")
        logger.info(f"  Noise Level: {health.noise_level:.4f}")
        
        if health.issues:
            logger.info(f"  Issues: {', '.join(health.issues)}")
        if health.warnings:
            logger.info(f"  Warnings: {', '.join(health.warnings)}")
    
    return checker


def demo_fallback_logging():
    """Demonstrate fallback logging"""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 2: Fallback Logging with Reason Codes")
    logger.info("=" * 80)
    
    checker = QuantumHealthChecker()
    
    # Simulate various fallback scenarios
    scenarios = [
        ("citation_traversal", FallbackReason.QUANTUM_NOISE_EXCEEDED, 
         "Noise level 0.15 exceeds threshold 0.10", 10),
        ("hypothesis_clustering", FallbackReason.INSUFFICIENT_QUBITS,
         "Required 20 qubits, only 10 available", 20),
        ("policy_optimization", FallbackReason.QUANTUM_TIMEOUT,
         "Execution exceeded 30s timeout", 5),
        ("citation_traversal", FallbackReason.QUANTUM_ERROR,
         "Circuit compilation failed", 8),
        ("hypothesis_clustering", FallbackReason.QUANTUM_RESOURCE_LIMIT,
         "Backend queue full", 15),
    ]
    
    logger.info("\nSimulating fallback events...")
    for operation, reason, details, qubits in scenarios:
        checker.log_fallback(
            operation=operation,
            reason=reason,
            reason_details=details,
            attempted_qubits=qubits,
            execution_time=np.random.uniform(0.5, 5.0)
        )
    
    # Get statistics
    stats = checker.get_fallback_statistics()
    
    logger.info(f"\nFallback Statistics:")
    logger.info(f"  Total fallbacks: {stats['total_fallbacks']}")
    logger.info(f"  Most common reason: {stats['most_common_reason']}")
    logger.info(f"  Avg time before fallback: {stats['avg_time_before_fallback']:.2f}s")
    
    logger.info(f"\nFallbacks by reason:")
    for reason, count in stats['reasons'].items():
        logger.info(f"  {reason}: {count}")
    
    logger.info(f"\nFallbacks by operation:")
    for operation, count in stats['operations'].items():
        logger.info(f"  {operation}: {count}")
    
    return checker


def demo_citation_walker_with_health():
    """Demonstrate citation walker with health monitoring"""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 3: Citation Walker with Health Monitoring")
    logger.info("=" * 80)
    
    # Generate test data
    generator = SyntheticDataGenerator(seed=42)
    corpus = generator.generate_synthetic_corpus(
        language="en",
        size=16,  # Requires 4 qubits
        domain="test",
        citation_density=0.15
    )
    
    # Create walker with health monitoring
    walker = QuantumCitationWalker(
        backend="qiskit_aer",
        shots=512,
        max_noise_threshold=0.1
    )
    
    logger.info(f"\nTraversing citation graph ({corpus.size} nodes)...")
    result = walker.traverse(
        adjacency_matrix=corpus.adjacency_matrix,
        semantic_weights=corpus.semantic_weights,
        start_nodes=[0, 1],
        max_steps=3
    )
    
    logger.info(f"Method used: {result['method']}")
    logger.info(f"Paths found: {len(result['paths'])}")
    
    if "quantum_health" in result:
        health = result["quantum_health"]
        logger.info(f"\nQuantum Health:")
        logger.info(f"  Readiness: {health['readiness_score']:.2f}")
        logger.info(f"  Noise: {health['noise_level']:.4f}")
        logger.info(f"  Qubits: {health['num_qubits']}")
    
    # Check fallback events
    fallback_stats = walker.health_checker.get_fallback_statistics()
    if fallback_stats['total_fallbacks'] > 0:
        logger.info(f"\nFallback Events: {fallback_stats['total_fallbacks']}")
        for reason, count in fallback_stats['reasons'].items():
            logger.info(f"  {reason}: {count}")
    
    return walker


def demo_evaluation_with_fallback_metrics():
    """Demonstrate evaluation harness with fallback metrics"""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 4: Evaluation with Fallback Metrics")
    logger.info("=" * 80)
    
    # Setup
    agent = MultilingualResearchAgent(
        supported_languages=[Language.ENGLISH],
        quantum_enabled=True
    )
    
    generator = SyntheticDataGenerator(seed=42)
    corpus_data = generator.generate_synthetic_corpus(
        language="en",
        size=12,
        domain="test",
        citation_density=0.15
    )
    
    agent.load_corpus(
        language=Language.ENGLISH,
        domain="test",
        corpus_data={
            "documents": corpus_data.documents,
            "citations": corpus_data.citations
        }
    )
    
    # Generate hypotheses
    hypotheses = []
    for i in range(10):
        hyp = Hypothesis(
            text=f"Test hypothesis {i}",
            language=Language.ENGLISH,
            confidence=0.8,
            embedding=np.random.randn(64)
        )
        hypotheses.append(hyp)
    
    # Run evaluation
    harness = EvaluationHarness(output_dir="evaluation_results")
    
    logger.info("\nRunning quantum pipeline...")
    quantum_metrics = harness.run_quantum_pipeline(agent, corpus_data, hypotheses)
    
    logger.info("\nRunning classical pipeline...")
    classical_metrics = harness.run_classical_pipeline(agent, corpus_data, hypotheses)
    
    # Compare
    comparison = harness.compare_results(quantum_metrics, classical_metrics)
    
    logger.info(f"\nComparison Results:")
    logger.info(f"  Winner: {comparison['winner']}")
    logger.info(f"  Quantum advantage: {comparison['quantum_advantage']:.4f}")
    
    # Show fallback metrics
    if quantum_metrics.fallback_metrics:
        fb = quantum_metrics.fallback_metrics
        logger.info(f"\nFallback Metrics:")
        logger.info(f"  Total fallbacks: {fb['total_fallbacks']}")
        logger.info(f"  Fallback rate: {fb['fallback_rate']:.2%}")
        if fb['most_common_reason']:
            logger.info(f"  Most common reason: {fb['most_common_reason']}")
        
        if fb['reasons']:
            logger.info(f"  Reasons breakdown:")
            for reason, count in fb['reasons'].items():
                logger.info(f"    {reason}: {count}")
    
    return harness


def demo_fallback_report():
    """Demonstrate comprehensive fallback reporting"""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 5: Comprehensive Fallback Report")
    logger.info("=" * 80)
    
    # Create harness with some fallback events
    harness = EvaluationHarness()
    
    if harness.health_checker:
        # Simulate some fallback events
        checker = harness.health_checker
        
        for i in range(5):
            checker.log_fallback(
                operation=f"operation_{i % 3}",
                reason=list(FallbackReason)[i % len(FallbackReason)],
                reason_details=f"Test fallback {i}",
                attempted_qubits=4 + i,
                execution_time=1.0 + i * 0.5
            )
        
        # Generate report
        report = harness.generate_fallback_report()
        
        logger.info(f"\nFallback Report:")
        logger.info(f"  Total events: {report['total_events']}")
        logger.info(f"  Total fallbacks: {report['overview']['total_fallbacks']}")
        
        logger.info(f"\nEvents by reason:")
        for reason, events in report['events_by_reason'].items():
            logger.info(f"  {reason}: {len(events)} events")
            for event in events[:2]:  # Show first 2
                logger.info(f"    - {event['operation']}: {event['details']}")
        
        logger.info(f"\nEvents by operation:")
        for operation, events in report['events_by_operation'].items():
            logger.info(f"  {operation}: {len(events)} events")
        
        logger.info(f"\nRecent timeline:")
        for event in report['timeline'][-5:]:
            logger.info(f"  {event['operation']} -> {event['reason']}: {event['details']}")
    else:
        logger.warning("Health checker not available")
    
    return harness


def demo_noise_threshold_testing():
    """Demonstrate noise threshold testing"""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 6: Noise Threshold Testing")
    logger.info("=" * 80)
    
    # Test different noise thresholds
    thresholds = [0.05, 0.1, 0.15, 0.2]
    
    for threshold in thresholds:
        logger.info(f"\nTesting with noise threshold: {threshold}")
        
        checker = QuantumHealthChecker(max_noise_threshold=threshold)
        health = checker.quantum_health_check("qiskit_aer", required_qubits=4)
        
        logger.info(f"  Readiness score: {health.readiness_score:.2f}")
        logger.info(f"  Available: {health.available}")
        
        if health.noise_level > threshold:
            logger.info(f"  ⚠️  Noise {health.noise_level:.4f} exceeds threshold {threshold}")
        else:
            logger.info(f"  ✓ Noise {health.noise_level:.4f} within threshold {threshold}")


def main():
    """Run all demos"""
    logger.info("=" * 80)
    logger.info("QUANTUM HEALTH MONITORING & FALLBACK TRACKING - DEMO")
    logger.info("=" * 80)
    
    try:
        # Demo 1: Health check
        demo_health_check()
        
        # Demo 2: Fallback logging
        demo_fallback_logging()
        
        # Demo 3: Citation walker with health
        demo_citation_walker_with_health()
        
        # Demo 4: Evaluation with fallback metrics
        demo_evaluation_with_fallback_metrics()
        
        # Demo 5: Fallback report
        demo_fallback_report()
        
        # Demo 6: Noise threshold testing
        demo_noise_threshold_testing()
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL DEMOS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        logger.info("\nKey Features Demonstrated:")
        logger.info("✓ Quantum health checking with readiness scores")
        logger.info("✓ Fallback logging with reason codes")
        logger.info("✓ Fallback metrics in evaluation harness")
        logger.info("✓ Comprehensive fallback reporting")
        logger.info("✓ Noise threshold testing")
        logger.info("✓ Resource limit detection")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
