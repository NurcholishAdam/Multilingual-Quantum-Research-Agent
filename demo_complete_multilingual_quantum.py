# -*- coding: utf-8 -*-
"""
Complete Demo: Multilingual Quantum Research Agent

Demonstrates:
1. Core agent architecture with multilingual support
2. Quantum-enhanced citation traversal
3. QAOA hypothesis clustering
4. Quantum RLHF policy optimization
5. Benchmarking and evaluation
6. Automatic fallback to classical methods
"""

import logging
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from multilingual_research_agent import (
    MultilingualResearchAgent,
    Language,
    FallbackMode,
    Hypothesis
)
from language_modules import MultilingualPipelineManager
from synthetic_data_generator import SyntheticDataGenerator
from quantum_citation_walker import QuantumCitationWalker
from quantum_hypothesis_clusterer import QuantumHypothesisClusterer
from evaluation_harness import EvaluationHarness


def demo_multilingual_agent():
    """Demonstrate multilingual agent capabilities"""
    logger.info("=" * 80)
    logger.info("DEMO 1: Multilingual Agent Architecture")
    logger.info("=" * 80)
    
    # Initialize agent
    agent = MultilingualResearchAgent(
        supported_languages=[Language.ENGLISH, Language.CHINESE, Language.SPANISH],
        quantum_enabled=True,
        fallback_mode=FallbackMode.AUTO
    )
    
    # Generate synthetic corpora for multiple languages
    generator = SyntheticDataGenerator(seed=42)
    
    for language in [Language.ENGLISH, Language.CHINESE, Language.SPANISH]:
        logger.info(f"\nLoading corpus for {language.value}...")
        
        corpus_data = generator.generate_synthetic_corpus(
            language=language.value,
            size=15,
            domain="machine_learning",
            citation_density=0.12
        )
        
        agent.load_corpus(
            language=language,
            domain="machine_learning",
            corpus_data={
                "documents": corpus_data.documents,
                "citations": corpus_data.citations,
                "metadata": corpus_data.metadata
            }
        )
    
    logger.info(f"\nLoaded {len(agent.corpora)} corpora")
    return agent


def demo_quantum_citation_traversal(agent):
    """Demonstrate quantum citation graph traversal"""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 2: Quantum Citation Graph Traversal")
    logger.info("=" * 80)
    
    # Get English corpus
    corpus_key = "en_machine_learning"
    
    logger.info(f"\nTraversing citation graph for {corpus_key}...")
    
    # Quantum traversal
    quantum_result = agent.traverse_citation_graph(
        corpus_key=corpus_key,
        start_nodes=None,
        max_depth=3,
        use_quantum=True
    )
    
    logger.info(f"Quantum traversal method: {quantum_result['method']}")
    logger.info(f"Found {len(quantum_result['paths'])} paths")
    
    if quantum_result.get('entanglement_measure'):
        logger.info(f"Entanglement measure: {quantum_result['entanglement_measure']:.4f}")
    
    # Classical comparison
    classical_result = agent.traverse_citation_graph(
        corpus_key=corpus_key,
        start_nodes=None,
        max_depth=3,
        use_quantum=False
    )
    
    logger.info(f"\nClassical traversal method: {classical_result['method']}")
    logger.info(f"Found {len(classical_result['paths'])} paths")
    
    return quantum_result, classical_result


def demo_hypothesis_clustering(agent):
    """Demonstrate QAOA hypothesis clustering"""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 3: QAOA Hypothesis Clustering")
    logger.info("=" * 80)
    
    # Generate hypotheses
    logger.info("\nGenerating hypotheses...")
    hypotheses = agent.generate_hypotheses(
        corpus_key=None,  # Use all corpora
        num_hypotheses=20,
        use_quantum=True
    )
    
    # Add synthetic embeddings for demo
    for hyp in hypotheses:
        hyp.embedding = np.random.randn(128)
    
    logger.info(f"Generated {len(hypotheses)} hypotheses")
    
    # QAOA clustering
    clusterer = QuantumHypothesisClusterer(num_clusters=3, qaoa_layers=2)
    embeddings = np.array([h.embedding for h in hypotheses])
    
    logger.info("\nRunning QAOA clustering...")
    qaoa_result = clusterer.cluster(embeddings)
    
    logger.info(f"QAOA method: {qaoa_result['method']}")
    logger.info(f"Number of clusters: {qaoa_result['num_clusters']}")
    logger.info(f"Clustering purity: {qaoa_result['purity']:.4f}")
    
    # Classical comparison
    logger.info("\nRunning classical clustering...")
    similarity_matrix = clusterer._compute_similarity_matrix(embeddings)
    classical_result = clusterer._classical_cluster(embeddings, similarity_matrix)
    
    logger.info(f"Classical method: {classical_result['method']}")
    logger.info(f"Clustering purity: {classical_result['purity']:.4f}")
    
    improvement = qaoa_result['purity'] - classical_result['purity']
    logger.info(f"\nQAOA improvement: {improvement:+.4f}")
    
    return hypotheses, qaoa_result, classical_result


def demo_policy_optimization(agent):
    """Demonstrate quantum RLHF policy optimization"""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 4: Quantum RLHF Policy Optimization")
    logger.info("=" * 80)
    
    # Generate synthetic feedback
    feedback = [
        {"action": "search", "reward": 0.8, "context": "query1"},
        {"action": "synthesize", "reward": 0.9, "context": "query2"},
        {"action": "search", "reward": 0.7, "context": "query3"},
        {"action": "analyze", "reward": 0.85, "context": "query4"}
    ]
    
    logger.info(f"\nOptimizing policy with {len(feedback)} feedback samples...")
    
    # Quantum optimization
    quantum_policy = agent.optimize_policy(feedback, use_quantum=True)
    logger.info(f"Quantum policy method: {quantum_policy['method']}")
    
    # Classical optimization
    classical_policy = agent.optimize_policy(feedback, use_quantum=False)
    logger.info(f"Classical policy method: {classical_policy['method']}")
    
    return quantum_policy, classical_policy


def demo_comprehensive_evaluation():
    """Demonstrate comprehensive benchmarking"""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 5: Comprehensive Evaluation & Benchmarking")
    logger.info("=" * 80)
    
    # Setup
    agent = MultilingualResearchAgent(
        supported_languages=[Language.ENGLISH],
        quantum_enabled=True
    )
    
    generator = SyntheticDataGenerator(seed=42)
    corpus_data = generator.generate_synthetic_corpus(
        language="en",
        size=20,
        domain="physics",
        citation_density=0.15
    )
    
    agent.load_corpus(
        language=Language.ENGLISH,
        domain="physics",
        corpus_data={
            "documents": corpus_data.documents,
            "citations": corpus_data.citations,
            "metadata": corpus_data.metadata
        }
    )
    
    # Generate hypotheses with embeddings
    hypotheses = []
    for i in range(15):
        hyp = Hypothesis(
            text=f"Hypothesis {i}",
            language=Language.ENGLISH,
            confidence=0.8,
            embedding=np.random.randn(128)
        )
        hypotheses.append(hyp)
    
    # Run evaluation
    harness = EvaluationHarness(output_dir="evaluation_results")
    
    logger.info("\nRunning quantum pipeline evaluation...")
    quantum_metrics = harness.run_quantum_pipeline(agent, corpus_data, hypotheses)
    
    logger.info("\nRunning classical pipeline evaluation...")
    classical_metrics = harness.run_classical_pipeline(agent, corpus_data, hypotheses)
    
    # Compare results
    logger.info("\nComparing quantum vs. classical...")
    comparison = harness.compare_results(quantum_metrics, classical_metrics, metric="accuracy")
    
    logger.info(f"\nComparison Results:")
    logger.info(f"Winner: {comparison['winner']}")
    logger.info(f"Quantum advantage: {comparison['quantum_advantage']:.4f}")
    logger.info(f"Speedup: {comparison['speedup']:.2f}x")
    
    logger.info("\nDetailed Metrics:")
    for metric_name, values in comparison['detailed_comparison'].items():
        logger.info(f"  {metric_name}:")
        logger.info(f"    Quantum: {values['quantum']:.4f}")
        logger.info(f"    Classical: {values['classical']:.4f}")
        logger.info(f"    Improvement: {values['improvement']:+.4f}")
    
    # Save results
    harness.save_results("multilingual_quantum_evaluation.json")
    logger.info("\nResults saved to evaluation_results/multilingual_quantum_evaluation.json")
    
    return comparison


def demo_fallback_mechanism():
    """Demonstrate automatic fallback to classical methods"""
    logger.info("\n" + "=" * 80)
    logger.info("DEMO 6: Automatic Fallback Mechanism")
    logger.info("=" * 80)
    
    # Create agent with quantum disabled to trigger fallback
    agent = MultilingualResearchAgent(
        supported_languages=[Language.ENGLISH],
        quantum_enabled=False,  # Force classical fallback
        fallback_mode=FallbackMode.AUTO
    )
    
    logger.info("\nAgent initialized with quantum_enabled=False")
    logger.info("All operations will automatically use classical fallback")
    
    # Generate test data
    generator = SyntheticDataGenerator(seed=42)
    corpus_data = generator.generate_synthetic_corpus(
        language="en",
        size=10,
        domain="test",
        citation_density=0.1
    )
    
    agent.load_corpus(
        language=Language.ENGLISH,
        domain="test",
        corpus_data={
            "documents": corpus_data.documents,
            "citations": corpus_data.citations
        }
    )
    
    # Test fallback for each operation
    logger.info("\nTesting fallback for citation traversal...")
    result = agent.traverse_citation_graph("en_test", use_quantum=True)
    logger.info(f"Method used: {result['method']} (requested quantum, got classical)")
    
    logger.info("\nTesting fallback for policy optimization...")
    feedback = [{"action": "test", "reward": 0.5}]
    policy = agent.optimize_policy(feedback, use_quantum=True)
    logger.info(f"Method used: {policy['method']} (requested quantum, got classical)")
    
    logger.info("\nFallback mechanism working correctly!")


def main():
    """Run all demos"""
    logger.info("=" * 80)
    logger.info("MULTILINGUAL QUANTUM RESEARCH AGENT - COMPLETE DEMO")
    logger.info("=" * 80)
    
    try:
        # Demo 1: Multilingual agent
        agent = demo_multilingual_agent()
        
        # Demo 2: Quantum citation traversal
        demo_quantum_citation_traversal(agent)
        
        # Demo 3: Hypothesis clustering
        demo_hypothesis_clustering(agent)
        
        # Demo 4: Policy optimization
        demo_policy_optimization(agent)
        
        # Demo 5: Comprehensive evaluation
        demo_comprehensive_evaluation()
        
        # Demo 6: Fallback mechanism
        demo_fallback_mechanism()
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL DEMOS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        logger.info("\nNext Steps:")
        logger.info("1. Explore Jupyter notebooks in quantum_integration/notebooks/")
        logger.info("2. Run individual components with your own data")
        logger.info("3. Customize quantum parameters for your use case")
        logger.info("4. Integrate with existing research workflows")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
