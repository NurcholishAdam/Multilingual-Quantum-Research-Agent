# Multilingual Quantum Research Agent - Quick Start Guide

## 🚀 5-Minute Setup

### 1. Install Dependencies

```bash
cd quantum_integration
pip install -r requirements.txt
```

### 2. Download Language Models (Optional)

```bash
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
python -m spacy download es_core_news_sm
```

### 3. Run Setup Script

```bash
python setup_multilingual_quantum.py
```

### 4. Test Installation

```bash
python test_multilingual_quantum_integration.py
```

## 📝 Basic Usage

### Example 1: Simple Agent

```python
from quantum_integration.multilingual_research_agent import (
    MultilingualResearchAgent,
    Language
)

# Create agent
agent = MultilingualResearchAgent(
    supported_languages=[Language.ENGLISH],
    quantum_enabled=True
)

# Load corpus
agent.load_corpus(
    language=Language.ENGLISH,
    domain="machine_learning",
    corpus_path="data/papers.json"
)

# Generate hypotheses
hypotheses = agent.generate_hypotheses(num_hypotheses=10)

print(f"Generated {len(hypotheses)} hypotheses")
```

### Example 2: Quantum Citation Traversal

```python
from quantum_integration.quantum_citation_walker import QuantumCitationWalker
import numpy as np

# Create citation network
adjacency = np.array([
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
], dtype=float)

semantic_weights = np.random.rand(4, 4)

# Run quantum walk
walker = QuantumCitationWalker()
result = walker.traverse(
    adjacency_matrix=adjacency,
    semantic_weights=semantic_weights,
    start_nodes=[0],
    max_steps=5
)

print(f"Method: {result['method']}")
print(f"Paths found: {len(result['paths'])}")
```

### Example 3: QAOA Clustering

```python
from quantum_integration.quantum_hypothesis_clusterer import QuantumHypothesisClusterer
import numpy as np

# Create hypothesis embeddings
embeddings = np.random.randn(20, 128)

# Run QAOA clustering
clusterer = QuantumHypothesisClusterer(num_clusters=3)
result = clusterer.cluster(embeddings)

print(f"Method: {result['method']}")
print(f"Clusters: {result['num_clusters']}")
print(f"Purity: {result['purity']:.3f}")
```

### Example 4: Synthetic Data Generation

```python
from quantum_integration.synthetic_data_generator import SyntheticDataGenerator

# Generate corpus
generator = SyntheticDataGenerator(seed=42)
corpus = generator.generate_synthetic_corpus(
    language="en",
    size=50,
    domain="physics",
    citation_density=0.15
)

print(f"Documents: {len(corpus.documents)}")
print(f"Citations: {len(corpus.citations)}")
```

### Example 5: Evaluation

```python
from quantum_integration.evaluation_harness import EvaluationHarness

# Setup
harness = EvaluationHarness()

# Run evaluations
quantum_metrics = harness.run_quantum_pipeline(agent, corpus, hypotheses)
classical_metrics = harness.run_classical_pipeline(agent, corpus, hypotheses)

# Compare
comparison = harness.compare_results(quantum_metrics, classical_metrics)

print(f"Winner: {comparison['winner']}")
print(f"Quantum advantage: {comparison['quantum_advantage']:.3f}")
```

## 📓 Jupyter Notebooks

### Citation Walk Demo

```bash
jupyter notebook notebooks/citation_walk_demo.ipynb
```

**What it does**:
- Loads synthetic citation network
- Runs quantum walk traversal
- Visualizes with NetworkX
- Compares with classical walk

### QAOA Clustering Demo

```bash
jupyter notebook notebooks/qaoa_clustering_demo.ipynb
```

**What it does**:
- Generates hypothesis embeddings
- Runs QAOA clustering
- Visualizes with PCA
- Compares with k-means

### Quantum RLHF Demo

```bash
jupyter notebook notebooks/quantum_rlhf_policy_demo.ipynb
```

**What it does**:
- Simulates agent feedback
- Optimizes policy with quantum RLHF
- Analyzes convergence
- Compares with classical RLHF

## 🎯 Complete Demo

Run all features at once:

```bash
python demo_complete_multilingual_quantum.py
```

This runs 6 demos:
1. Multilingual agent architecture
2. Quantum citation traversal
3. QAOA hypothesis clustering
4. Quantum RLHF policy optimization
5. Comprehensive evaluation
6. Automatic fallback mechanism

## ⚙️ Configuration

### Quantum Backend

```python
# Use Qiskit Aer (default)
agent = MultilingualResearchAgent(quantum_enabled=True)

# Disable quantum (classical only)
agent = MultilingualResearchAgent(quantum_enabled=False)
```

### Fallback Mode

```python
from quantum_integration.multilingual_research_agent import FallbackMode

# Automatic fallback on errors
agent = MultilingualResearchAgent(fallback_mode=FallbackMode.AUTO)

# Manual control
agent = MultilingualResearchAgent(fallback_mode=FallbackMode.MANUAL)
```

### Language Selection

```python
from quantum_integration.multilingual_research_agent import Language

# Single language
agent = MultilingualResearchAgent(
    supported_languages=[Language.ENGLISH]
)

# Multiple languages
agent = MultilingualResearchAgent(
    supported_languages=[
        Language.ENGLISH,
        Language.CHINESE,
        Language.SPANISH
    ]
)
```

## 🔧 Troubleshooting

### Qiskit Not Available

If quantum features fail:
```bash
pip install qiskit qiskit-aer qiskit-algorithms
```

Agent will automatically fallback to classical methods.

### spaCy Models Missing

If language processing fails:
```bash
python -m spacy download en_core_web_sm
```

Or use the setup script:
```bash
python setup_multilingual_quantum.py
```

### Import Errors

Make sure you're in the right directory:
```bash
cd quantum_integration
python -c "from multilingual_research_agent import MultilingualResearchAgent"
```

### Memory Issues

For large corpora, reduce size:
```python
corpus = generator.generate_synthetic_corpus(
    language="en",
    size=20,  # Smaller size
    domain="test"
)
```

## 📚 Next Steps

1. **Read Full Documentation**: [MULTILINGUAL_QUANTUM_README.md](MULTILINGUAL_QUANTUM_README.md)
2. **Check Completion Summary**: [MULTILINGUAL_QUANTUM_COMPLETION_SUMMARY.md](MULTILINGUAL_QUANTUM_COMPLETION_SUMMARY.md)
3. **Explore Notebooks**: `notebooks/` directory
4. **Run Tests**: `python test_multilingual_quantum_integration.py`
5. **Customize**: Adapt for your research domain

## 🤝 Getting Help

- **Issues**: Check error messages and logs
- **Documentation**: Read the full README
- **Examples**: Look at demo scripts and notebooks
- **Tests**: Run integration tests to validate setup

## 📊 Performance Tips

### For Speed
- Use `quantum_enabled=False` for faster classical-only mode
- Reduce corpus size for testing
- Lower `shots` parameter in quantum modules

### For Accuracy
- Increase `qaoa_layers` for better clustering
- Use more `shots` for quantum circuits
- Increase corpus size for better statistics

### For Memory
- Process corpora in batches
- Use smaller embedding dimensions
- Clear unused corpora from agent

## 🎓 Learning Resources

### Quantum Computing
- [Qiskit Textbook](https://qiskit.org/textbook/)
- [QAOA Tutorial](https://qiskit.org/textbook/ch-applications/qaoa.html)
- [Quantum Walks](https://arxiv.org/abs/quant-ph/0012090)

### NLP
- [spaCy Documentation](https://spacy.io/)
- [Transformers Guide](https://huggingface.co/docs/transformers/)
- [Multilingual NLP](https://arxiv.org/abs/2004.09813)

### Research Agent
- [LIMIT-GRAPH Paper](../extensions/LIMIT-GRAPH/)
- [DCoT Agent Aligner](../extensions/LIMIT-GRAPH/DCoTAgentAligner/)
- [Social Science Extensions](social_science_extensions/)

---

**Ready to start?** Run `python demo_complete_multilingual_quantum.py` 🚀
