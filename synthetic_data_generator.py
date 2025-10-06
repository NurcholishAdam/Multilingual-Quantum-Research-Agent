# -*- coding: utf-8 -*-
"""
Synthetic Dataset Generator for Multilingual Research Agent

Generates synthetic corpora with:
- Multilingual abstracts
- Citation networks
- Norm emergence patterns
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SyntheticCorpus:
    """Container for synthetic research corpus"""
    language: str
    domain: str
    size: int
    documents: List[Dict[str, Any]]
    citations: List[tuple]
    adjacency_matrix: np.ndarray
    semantic_weights: np.ndarray
    metadata: Dict[str, Any]


class SyntheticDataGenerator:
    """Generate synthetic multilingual research corpora"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Template texts for different languages
        self.templates = {
            "en": [
                "This research investigates {topic} in the context of {domain}.",
                "We propose a novel approach to {topic} using {method}.",
                "Our findings suggest that {topic} is influenced by {factor}."
            ],
            "id": [
                "Penelitian ini menyelidiki {topic} dalam konteks {domain}.",
                "Kami mengusulkan pendekatan baru untuk {topic} menggunakan {method}.",
                "Temuan kami menunjukkan bahwa {topic} dipengaruhi oleh {factor}."
            ],
            "zh": [
                "本研究调查了{domain}背景下的{topic}。",
                "我们提出了一种使用{method}的{topic}新方法。",
                "我们的研究结果表明{topic}受{factor}影响。"
            ],
            "ar": [
                "يبحث هذا البحث في {topic} في سياق {domain}.",
                "نقترح نهجًا جديدًا لـ {topic} باستخدام {method}.",
                "تشير نتائجنا إلى أن {topic} يتأثر بـ {factor}."
            ],
            "es": [
                "Esta investigación examina {topic} en el contexto de {domain}.",
                "Proponemos un enfoque novedoso para {topic} utilizando {method}.",
                "Nuestros hallazgos sugieren que {topic} está influenciado por {factor}."
            ]
        }
        
        self.topics = ["quantum computing", "machine learning", "social networks", "climate change"]
        self.methods = ["neural networks", "statistical analysis", "simulation", "optimization"]
        self.factors = ["data quality", "model complexity", "environmental conditions"]
    
    def generate_synthetic_corpus(
        self,
        language: str,
        size: int,
        domain: str,
        citation_density: float = 0.1,
        include_norms: bool = True
    ) -> SyntheticCorpus:
        """
        Generate synthetic research corpus.
        
        Args:
            language: Language code (en, id, zh, ar, es)
            size: Number of documents to generate
            domain: Research domain
            citation_density: Probability of citation between documents
            include_norms: Whether to include norm emergence patterns
        
        Returns:
            SyntheticCorpus object
        """
        logger.info(f"Generating synthetic corpus: language={language}, size={size}, domain={domain}")
        
        # Generate documents
        documents = self._generate_documents(language, size, domain)
        
        # Generate citation network
        citations, adjacency_matrix = self._generate_citation_network(size, citation_density)
        
        # Generate semantic weights
        semantic_weights = self._generate_semantic_weights(size, documents)
        
        # Generate norm patterns if requested
        metadata = {}
        if include_norms:
            metadata["norms"] = self._generate_norm_patterns(size)
        
        metadata.update({
            "language": language,
            "domain": domain,
            "size": size,
            "citation_density": citation_density,
            "generation_seed": self.seed
        })
        
        corpus = SyntheticCorpus(
            language=language,
            domain=domain,
            size=size,
            documents=documents,
            citations=citations,
            adjacency_matrix=adjacency_matrix,
            semantic_weights=semantic_weights,
            metadata=metadata
        )
        
        logger.info(f"Generated corpus with {len(documents)} documents and {len(citations)} citations")
        return corpus
    
    def _generate_documents(
        self,
        language: str,
        size: int,
        domain: str
    ) -> List[Dict[str, Any]]:
        """Generate synthetic documents"""
        documents = []
        templates = self.templates.get(language, self.templates["en"])
        
        for i in range(size):
            template = np.random.choice(templates)
            text = template.format(
                topic=np.random.choice(self.topics),
                domain=domain,
                method=np.random.choice(self.methods),
                factor=np.random.choice(self.factors)
            )
            
            doc = {
                "id": f"doc_{i}",
                "title": f"Research Paper {i}",
                "abstract": text,
                "language": language,
                "domain": domain,
                "year": 2020 + (i % 5),
                "embedding": np.random.randn(128)  # Synthetic embedding
            }
            documents.append(doc)
        
        return documents
    
    def _generate_citation_network(
        self,
        size: int,
        density: float
    ) -> tuple:
        """Generate citation network"""
        adjacency_matrix = np.zeros((size, size))
        citations = []
        
        # Generate citations (directed graph, older papers cite newer)
        for i in range(size):
            for j in range(i + 1, size):
                if np.random.random() < density:
                    adjacency_matrix[j, i] = 1  # j cites i
                    citations.append((j, i))
        
        return citations, adjacency_matrix
    
    def _generate_semantic_weights(
        self,
        size: int,
        documents: List[Dict]
    ) -> np.ndarray:
        """Generate semantic similarity weights"""
        # Use document embeddings to compute similarity
        embeddings = np.array([doc["embedding"] for doc in documents])
        
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(normalized, normalized.T)
        
        # Ensure non-negative and scale to [0, 1]
        similarity = (similarity + 1) / 2
        
        return similarity
    
    def _generate_norm_patterns(self, size: int) -> Dict[str, Any]:
        """Generate norm emergence patterns"""
        # Simulate social norms evolving over time
        time_steps = 10
        norms = []
        
        for t in range(time_steps):
            # Norm strength increases over time with some noise
            strength = min(1.0, 0.1 * t + np.random.normal(0, 0.1))
            adoption_rate = np.random.beta(2, 5, size)  # Adoption across documents
            
            norms.append({
                "time_step": t,
                "norm_strength": max(0, strength),
                "adoption_rates": adoption_rate.tolist(),
                "consensus": np.mean(adoption_rate)
            })
        
        return {
            "time_steps": time_steps,
            "evolution": norms,
            "final_consensus": norms[-1]["consensus"]
        }
    
    def generate_multilingual_corpus_set(
        self,
        languages: List[str],
        size_per_language: int,
        domain: str
    ) -> Dict[str, SyntheticCorpus]:
        """Generate corpora for multiple languages"""
        corpora = {}
        
        for language in languages:
            corpus = self.generate_synthetic_corpus(
                language=language,
                size=size_per_language,
                domain=domain
            )
            corpora[language] = corpus
        
        logger.info(f"Generated {len(corpora)} multilingual corpora")
        return corpora
