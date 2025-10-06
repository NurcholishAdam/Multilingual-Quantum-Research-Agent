# -*- coding: utf-8 -*-
"""
Integration Test for Multilingual Quantum Research Agent

Quick validation of all components.
"""

import sys
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_agent_initialization():
    """Test agent initialization"""
    logger.info("TEST 1: Agent Initialization")
    
    from multilingual_research_agent import (
        MultilingualResearchAgent,
        Language,
        FallbackMode
    )
    
    agent = MultilingualResearchAgent(
        supported_languages=[Language.ENGLISH, Language.CHINESE],
        quantum_enabled=True,
        fallback_mode=FallbackMode.AUTO
    )
    
    assert agent is not None
    assert len(agent.supported_languages) == 2
    logger.info("✓ Agent initialization successful\n")
    return agent


def test_synthetic_data_generation():
    """Test synthetic data generator"""
    logger.info("TEST 2: Synthetic Data Generation")
    
    from synthetic_data_generator import SyntheticDataGenerator
    
    generator = SyntheticDataGenerator(seed=42)
    corpus = generator.generate_synthetic_corpus(
        language="en",
        size=10,
        domain="test",
        citation_density=0.1
    )
    
    assert corpus.size == 10
    assert len(corpus.documents) == 10
    assert corpus.adjacency_matrix.shape == (10, 10)
    logger.info(f"✓ Generated corpus with {corpus.size} documents\n")
    return corpus


def test_language_pipeline():
    """Test language pipeline"""
    logger.info("TEST 3: Language Pipeline")
    
    from language_modules import MultilingualPipelineManager
    
    manager = MultilingualPipelineManager()
    pipeline = manager.get_pipeline("en")
    
    text = "This is a test sentence for NLP processing."
    tokens = pipeline.tokenize(text)
    
    assert len(tokens) > 0
    logger.info(f"✓ Tokenized text into {len(tokens)} tokens\n")
    return manager


def test_quantum_citation_walker():
    """Test quantum citation walker"""
    logger.info("TEST 4: Quantum Citation Walker")
    
    from quantum_citation_walker import QuantumCitationWalker
    
    walker = QuantumCitationWalker(shots=512)
    
    # Create small test graph
    adjacency = np.array([
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype=float)
    
    semantic_weights = np.random.rand(4, 4)
    
    result = walker.traverse(
        adjacency_matrix=adjacency,
        semantic_weights=semantic_weights,
        start_nodes=[0],
        max_steps=3
    )
    
    assert "method" in result
    assert "paths" in result
    logger.info(f"✓ Traversal completed using {result['method']} method\n")
    return result


def test_quantum_hypothesis_clusterer():
    """Test quantum hypothesis clusterer"""
    logger.info("TEST 5: Quantum Hypothesis Clusterer")
    
    from quantum_hypothesis_clusterer import QuantumHypothesisClusterer
    
    clusterer = QuantumHypothesisClusterer(num_clusters=2, qaoa_layers=1)
    
    # Create test embeddings
    embeddings = np.random.randn(10, 64)
    
    result = clusterer.cluster(embeddings)
    
    assert "method" in result
    assert "cluster_assignments" in result
    assert "purity" in result
    logger.info(f"✓ Clustering completed using {result['method']} method\n")
    return result


def test_evaluation_harness():
    """Test evaluation harness"""
    logger.info("TEST 6: Evaluation Harness")
    
    from evaluation_harness import EvaluationHarness, EvaluationMetrics
    
    harness = EvaluationHarness(output_dir="test_results")
    
    # Create mock metrics
    quantum_metrics = EvaluationMetrics(
        traversal_efficiency=0.85,
        clustering_purity=0.78,
        rlhf_convergence=0.82,
        execution_time=2.3,
        method="quantum",
        additional_metrics={}
    )
    
    classical_metrics = EvaluationMetrics(
        traversal_efficiency=0.72,
        clustering_purity=0.71,
        rlhf_convergence=0.75,
        execution_time=1.8,
        method="classical",
        additional_metrics={}
    )
    
    comparison = harness.compare_results(quantum_metrics, classical_metrics)
    
    assert "quantum_advantage" in comparison
    assert "winner" in comparison
    logger.info(f"✓ Comparison complete: {comparison['winner']} wins\n")
    return comparison


def test_end_to_end_workflow():
    """Test complete end-to-end workflow"""
    logger.info("TEST 7: End-to-End Workflow")
    
    from multilingual_research_agent import (
        MultilingualResearchAgent,
        Language,
        Hypothesis
    )
    from synthetic_data_generator import SyntheticDataGenerator
    
    # Initialize
    agent = MultilingualResearchAgent(
        supported_languages=[Language.ENGLISH],
        quantum_enabled=True
    )
    
    # Generate data
    generator = SyntheticDataGenerator(seed=42)
    corpus_data = generator.generate_synthetic_corpus(
        language="en",
        size=8,
        domain="test",
        citation_density=0.15
    )
    
    # Load corpus
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
    for i in range(5):
        hyp = Hypothesis(
            text=f"Test hypothesis {i}",
            language=Language.ENGLISH,
            confidence=0.8,
            embedding=np.random.randn(64)
        )
        hypotheses.append(hyp)
    
    # Traverse citations
    traversal_result = agent.traverse_citation_graph(
        corpus_key="en_test",
        max_depth=2
    )
    
    # Optimize policy
    feedback = [{"action": "test", "reward": 0.8}]
    policy_result = agent.optimize_policy(feedback)
    
    assert len(agent.corpora) > 0
    assert traversal_result is not None
    assert policy_result is not None
    logger.info("✓ End-to-end workflow successful\n")
    return True


def run_all_tests():
    """Run all integration tests"""
    logger.info("=" * 70)
    logger.info("MULTILINGUAL QUANTUM RESEARCH AGENT - INTEGRATION TESTS")
    logger.info("=" * 70)
    logger.info("")
    
    tests = [
        ("Agent Initialization", test_agent_initialization),
        ("Synthetic Data Generation", test_synthetic_data_generation),
        ("Language Pipeline", test_language_pipeline),
        ("Quantum Citation Walker", test_quantum_citation_walker),
        ("Quantum Hypothesis Clusterer", test_quantum_hypothesis_clusterer),
        ("Evaluation Harness", test_evaluation_harness),
        ("End-to-End Workflow", test_end_to_end_workflow)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED: {e}")
            failed += 1
    
    logger.info("=" * 70)
    logger.info(f"TEST RESULTS: {passed} passed, {failed} failed")
    logger.info("=" * 70)
    
    if failed == 0:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("\nThe multilingual quantum research agent is ready to use.")
        logger.info("Run 'python demo_complete_multilingual_quantum.py' for full demo.")
        return True
    else:
        logger.error(f"\n❌ {failed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
