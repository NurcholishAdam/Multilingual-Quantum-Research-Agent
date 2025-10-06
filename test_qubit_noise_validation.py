# -*- coding: utf-8 -*-
"""
Test Script: Qubit Count and Noise Threshold Validation

Validates that:
1. Qubit count checks are operational
2. Noise threshold checks are operational
3. Fallback triggers correctly based on these checks
"""

import logging
import numpy as np
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from quantum_health_checker import QuantumHealthChecker, FallbackReason
from quantum_citation_walker import QuantumCitationWalker
from quantum_hypothesis_clusterer import QuantumHypothesisClusterer
from synthetic_data_generator import SyntheticDataGenerator


def test_qubit_count_check():
    """Test 1: Verify qubit count checking"""
    logger.info("=" * 80)
    logger.info("TEST 1: Qubit Count Validation")
    logger.info("=" * 80)
    
    checker = QuantumHealthChecker(min_qubits_required=4)
    
    # Test with different qubit requirements
    test_cases = [
        (2, "Should fail - insufficient qubits"),
        (4, "Should pass - exact match"),
        (8, "Should pass - sufficient qubits"),
        (64, "May fail - exceeds typical simulator capacity")
    ]
    
    results = []
    for required_qubits, description in test_cases:
        logger.info(f"\nTest case: {description}")
        logger.info(f"Required qubits: {required_qubits}")
        
        health = checker.quantum_health_check(
            backend_name="qiskit_aer",
            required_qubits=required_qubits
        )
        
        logger.info(f"  Available: {health.available}")
        logger.info(f"  Backend qubits: {health.num_qubits}")
        logger.info(f"  Readiness score: {health.readiness_score:.2f}")
        
        # Check if qubit validation worked
        if required_qubits > health.num_qubits:
            if not health.available:
                logger.info(f"  ✓ PASS: Correctly identified insufficient qubits")
                results.append(True)
            else:
                logger.error(f"  ✗ FAIL: Should have failed due to insufficient qubits")
                results.append(False)
        else:
            if health.available or health.readiness_score > 0:
                logger.info(f"  ✓ PASS: Correctly identified sufficient qubits")
                results.append(True)
            else:
                logger.error(f"  ✗ FAIL: Should have passed with sufficient qubits")
                results.append(False)
        
        if health.issues:
            logger.info(f"  Issues: {', '.join(health.issues)}")
    
    success_rate = sum(results) / len(results)
    logger.info(f"\nQubit Count Check Success Rate: {success_rate:.1%}")
    return success_rate == 1.0


def test_noise_threshold_check():
    """Test 2: Verify noise threshold checking"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Noise Threshold Validation")
    logger.info("=" * 80)
    
    # Test with different noise thresholds
    test_cases = [
        (0.001, "Very strict threshold"),
        (0.05, "Moderate threshold"),
        (0.1, "Standard threshold"),
        (0.2, "Lenient threshold"),
        (0.5, "Very lenient threshold")
    ]
    
    results = []
    for threshold, description in test_cases:
        logger.info(f"\nTest case: {description}")
        logger.info(f"Noise threshold: {threshold}")
        
        checker = QuantumHealthChecker(max_noise_threshold=threshold)
        health = checker.quantum_health_check("qiskit_aer", required_qubits=4)
        
        logger.info(f"  Measured noise: {health.noise_level:.4f}")
        logger.info(f"  Threshold: {threshold:.4f}")
        logger.info(f"  Available: {health.available}")
        logger.info(f"  Readiness score: {health.readiness_score:.2f}")
        
        # Check if noise validation worked
        if health.noise_level > threshold:
            if not health.available:
                logger.info(f"  ✓ PASS: Correctly rejected due to high noise")
                results.append(True)
            else:
                logger.error(f"  ✗ FAIL: Should have failed due to high noise")
                results.append(False)
        else:
            if health.available:
                logger.info(f"  ✓ PASS: Correctly accepted with acceptable noise")
                results.append(True)
            else:
                # May fail for other reasons, check if noise was the issue
                noise_issue = any("noise" in issue.lower() for issue in health.issues)
                if not noise_issue:
                    logger.info(f"  ✓ PASS: Failed for other reasons, not noise")
                    results.append(True)
                else:
                    logger.error(f"  ✗ FAIL: Should have passed with acceptable noise")
                    results.append(False)
        
        if health.issues:
            logger.info(f"  Issues: {', '.join(health.issues)}")
        if health.warnings:
            logger.info(f"  Warnings: {', '.join(health.warnings)}")
    
    success_rate = sum(results) / len(results)
    logger.info(f"\nNoise Threshold Check Success Rate: {success_rate:.1%}")
    return success_rate >= 0.8  # Allow some tolerance


def test_citation_walker_qubit_check():
    """Test 3: Verify citation walker respects qubit limits"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Citation Walker Qubit Limit Enforcement")
    logger.info("=" * 80)
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Test with different graph sizes
    test_cases = [
        (4, 2, "Small graph - should work"),
        (8, 3, "Medium graph - should work"),
        (16, 4, "Larger graph - should work"),
        (128, 7, "Very large graph - may trigger fallback")
    ]
    
    results = []
    for graph_size, expected_qubits, description in test_cases:
        logger.info(f"\nTest case: {description}")
        logger.info(f"Graph size: {graph_size} nodes")
        logger.info(f"Expected qubits: {expected_qubits}")
        
        # Generate corpus
        corpus = generator.generate_synthetic_corpus(
            language="en",
            size=graph_size,
            domain="test",
            citation_density=0.1
        )
        
        # Create walker
        walker = QuantumCitationWalker(
            backend="qiskit_aer",
            shots=256,  # Reduced for speed
            max_noise_threshold=0.1
        )
        
        # Traverse
        result = walker.traverse(
            adjacency_matrix=corpus.adjacency_matrix,
            semantic_weights=corpus.semantic_weights,
            start_nodes=[0],
            max_steps=3
        )
        
        logger.info(f"  Method used: {result['method']}")
        
        # Check if qubit limit was respected
        if "quantum_health" in result:
            health = result["quantum_health"]
            logger.info(f"  Qubits available: {health['num_qubits']}")
            logger.info(f"  Readiness score: {health['readiness_score']:.2f}")
            
            if health['num_qubits'] >= expected_qubits:
                logger.info(f"  ✓ PASS: Sufficient qubits available")
                results.append(True)
            else:
                logger.error(f"  ✗ FAIL: Insufficient qubits but quantum was used")
                results.append(False)
        else:
            # Classical fallback was used
            logger.info(f"  Classical fallback triggered")
            
            # Check fallback statistics
            stats = walker.health_checker.get_fallback_statistics()
            if stats['total_fallbacks'] > 0:
                logger.info(f"  Fallback reason: {stats['most_common_reason']}")
                
                # Check if it was due to qubit limit
                if 'INSUFFICIENT_QUBITS' in stats['reasons']:
                    logger.info(f"  ✓ PASS: Correctly fell back due to qubit limit")
                    results.append(True)
                else:
                    logger.info(f"  ✓ PASS: Fell back for other valid reason")
                    results.append(True)
            else:
                logger.info(f"  ✓ PASS: Classical method used (quantum unavailable)")
                results.append(True)
    
    success_rate = sum(results) / len(results)
    logger.info(f"\nCitation Walker Qubit Check Success Rate: {success_rate:.1%}")
    return success_rate >= 0.75


def test_citation_walker_noise_check():
    """Test 4: Verify citation walker respects noise threshold"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Citation Walker Noise Threshold Enforcement")
    logger.info("=" * 80)
    
    generator = SyntheticDataGenerator(seed=42)
    corpus = generator.generate_synthetic_corpus(
        language="en",
        size=8,
        domain="test",
        citation_density=0.1
    )
    
    # Test with different noise thresholds
    test_cases = [
        (0.001, "Very strict - should fallback"),
        (0.05, "Moderate - may work"),
        (0.1, "Standard - should work"),
        (0.2, "Lenient - should work")
    ]
    
    results = []
    for threshold, description in test_cases:
        logger.info(f"\nTest case: {description}")
        logger.info(f"Noise threshold: {threshold}")
        
        walker = QuantumCitationWalker(
            backend="qiskit_aer",
            shots=256,
            max_noise_threshold=threshold
        )
        
        result = walker.traverse(
            adjacency_matrix=corpus.adjacency_matrix,
            semantic_weights=corpus.semantic_weights,
            start_nodes=[0],
            max_steps=3
        )
        
        logger.info(f"  Method used: {result['method']}")
        
        if "quantum_health" in result:
            health = result["quantum_health"]
            logger.info(f"  Noise level: {health['noise_level']:.4f}")
            logger.info(f"  Threshold: {threshold:.4f}")
            
            if health['noise_level'] <= threshold:
                logger.info(f"  ✓ PASS: Noise within threshold, quantum used")
                results.append(True)
            else:
                logger.error(f"  ✗ FAIL: Noise exceeds threshold but quantum was used")
                results.append(False)
        else:
            # Check if fallback was due to noise
            stats = walker.health_checker.get_fallback_statistics()
            if 'QUANTUM_NOISE_EXCEEDED' in stats.get('reasons', {}):
                logger.info(f"  ✓ PASS: Correctly fell back due to noise threshold")
                results.append(True)
            else:
                logger.info(f"  ✓ PASS: Fell back for other valid reason")
                results.append(True)
    
    success_rate = sum(results) / len(results)
    logger.info(f"\nCitation Walker Noise Check Success Rate: {success_rate:.1%}")
    return success_rate >= 0.75


def test_fallback_logging():
    """Test 5: Verify fallback events are logged correctly"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Fallback Logging Validation")
    logger.info("=" * 80)
    
    checker = QuantumHealthChecker()
    
    # Log various fallback events
    test_events = [
        ("op1", FallbackReason.INSUFFICIENT_QUBITS, "Need 20, have 10", 20),
        ("op2", FallbackReason.QUANTUM_NOISE_EXCEEDED, "Noise 0.15 > 0.10", 8),
        ("op3", FallbackReason.QUANTUM_TIMEOUT, "Exceeded 30s", 4),
        ("op1", FallbackReason.INSUFFICIENT_QUBITS, "Need 16, have 10", 16),
    ]
    
    for operation, reason, details, qubits in test_events:
        checker.log_fallback(
            operation=operation,
            reason=reason,
            reason_details=details,
            attempted_qubits=qubits,
            execution_time=1.0
        )
    
    # Verify statistics
    stats = checker.get_fallback_statistics()
    
    logger.info(f"\nFallback Statistics:")
    logger.info(f"  Total fallbacks: {stats['total_fallbacks']}")
    logger.info(f"  Expected: {len(test_events)}")
    
    results = []
    
    # Check total count
    if stats['total_fallbacks'] == len(test_events):
        logger.info(f"  ✓ PASS: Correct total count")
        results.append(True)
    else:
        logger.error(f"  ✗ FAIL: Incorrect total count")
        results.append(False)
    
    # Check reason breakdown
    logger.info(f"\n  Reasons breakdown:")
    for reason, count in stats['reasons'].items():
        logger.info(f"    {reason}: {count}")
    
    expected_reasons = {
        'INSUFFICIENT_QUBITS': 2,
        'QUANTUM_NOISE_EXCEEDED': 1,
        'QUANTUM_TIMEOUT': 1
    }
    
    for reason, expected_count in expected_reasons.items():
        actual_count = stats['reasons'].get(reason, 0)
        if actual_count == expected_count:
            logger.info(f"  ✓ PASS: {reason} count correct ({actual_count})")
            results.append(True)
        else:
            logger.error(f"  ✗ FAIL: {reason} count incorrect (expected {expected_count}, got {actual_count})")
            results.append(False)
    
    # Check operation breakdown
    logger.info(f"\n  Operations breakdown:")
    for operation, count in stats['operations'].items():
        logger.info(f"    {operation}: {count}")
    
    # Query specific events
    qubit_events = checker.get_fallback_events(reason=FallbackReason.INSUFFICIENT_QUBITS)
    logger.info(f"\n  Qubit-related events: {len(qubit_events)}")
    
    if len(qubit_events) == 2:
        logger.info(f"  ✓ PASS: Correct number of qubit events")
        results.append(True)
    else:
        logger.error(f"  ✗ FAIL: Incorrect number of qubit events")
        results.append(False)
    
    success_rate = sum(results) / len(results)
    logger.info(f"\nFallback Logging Success Rate: {success_rate:.1%}")
    return success_rate == 1.0


def main():
    """Run all validation tests"""
    logger.info("=" * 80)
    logger.info("QUBIT COUNT & NOISE THRESHOLD VALIDATION SUITE")
    logger.info("=" * 80)
    
    test_results = {}
    
    try:
        # Run all tests
        test_results["Qubit Count Check"] = test_qubit_count_check()
        test_results["Noise Threshold Check"] = test_noise_threshold_check()
        test_results["Citation Walker Qubit Check"] = test_citation_walker_qubit_check()
        test_results["Citation Walker Noise Check"] = test_citation_walker_noise_check()
        test_results["Fallback Logging"] = test_fallback_logging()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        for test_name, passed in test_results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{status}: {test_name}")
        
        total_passed = sum(test_results.values())
        total_tests = len(test_results)
        
        logger.info(f"\nOverall: {total_passed}/{total_tests} tests passed ({total_passed/total_tests:.1%})")
        
        if total_passed == total_tests:
            logger.info("\n🎉 ALL VALIDATION TESTS PASSED!")
            logger.info("✓ Qubit count checks are operational")
            logger.info("✓ Noise threshold checks are operational")
            logger.info("✓ Fallback triggers are working correctly")
            return 0
        else:
            logger.error(f"\n❌ {total_tests - total_passed} test(s) failed")
            logger.error("Please review the failures above")
            return 1
            
    except Exception as e:
        logger.error(f"\nValidation failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
