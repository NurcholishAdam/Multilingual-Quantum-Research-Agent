# -*- coding: utf-8 -*-
"""
Setup Script for Multilingual Quantum Research Agent

Handles:
- Dependency installation
- spaCy model downloads
- Directory structure creation
- Configuration validation
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version >= 3.8"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_requirements():
    """Install Python dependencies"""
    logger.info("Installing Python dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False


def download_spacy_models():
    """Download spaCy language models"""
    models = [
        ("en_core_web_sm", "English"),
        ("zh_core_web_sm", "Chinese"),
        ("es_core_news_sm", "Spanish")
    ]
    
    logger.info("Downloading spaCy models...")
    success = True
    
    for model, language in models:
        try:
            logger.info(f"  Downloading {language} model ({model})...")
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", model
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"  ✓ {language} model installed")
        except subprocess.CalledProcessError:
            logger.warning(f"  ✗ Failed to download {language} model (optional)")
            success = False
    
    return success


def create_directories():
    """Create necessary directories"""
    directories = [
        "evaluation_results",
        "notebooks",
        "data",
        "logs"
    ]
    
    logger.info("Creating directories...")
    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        logger.info(f"  ✓ {directory}/")
    
    return True


def check_quantum_availability():
    """Check if quantum libraries are available"""
    logger.info("Checking quantum computing libraries...")
    
    try:
        import qiskit
        logger.info(f"  ✓ Qiskit {qiskit.__version__}")
    except ImportError:
        logger.warning("  ✗ Qiskit not available")
        return False
    
    try:
        from qiskit_aer import AerSimulator
        logger.info("  ✓ Qiskit Aer available")
    except ImportError:
        logger.warning("  ✗ Qiskit Aer not available")
        return False
    
    return True


def check_nlp_availability():
    """Check if NLP libraries are available"""
    logger.info("Checking NLP libraries...")
    
    try:
        import spacy
        logger.info(f"  ✓ spaCy {spacy.__version__}")
    except ImportError:
        logger.warning("  ✗ spaCy not available")
        return False
    
    try:
        import transformers
        logger.info(f"  ✓ Transformers {transformers.__version__}")
    except ImportError:
        logger.warning("  ✗ Transformers not available")
        return False
    
    return True


def run_basic_test():
    """Run basic functionality test"""
    logger.info("Running basic functionality test...")
    
    try:
        from multilingual_research_agent import MultilingualResearchAgent, Language
        
        agent = MultilingualResearchAgent(
            supported_languages=[Language.ENGLISH],
            quantum_enabled=False  # Use classical for test
        )
        
        logger.info("  ✓ Agent initialization successful")
        return True
    except Exception as e:
        logger.error(f"  ✗ Agent initialization failed: {e}")
        return False


def print_next_steps():
    """Print next steps for user"""
    logger.info("\n" + "=" * 70)
    logger.info("SETUP COMPLETE!")
    logger.info("=" * 70)
    
    logger.info("\n📚 Next Steps:")
    logger.info("  1. Run the complete demo:")
    logger.info("     python demo_complete_multilingual_quantum.py")
    logger.info("\n  2. Explore Jupyter notebooks:")
    logger.info("     jupyter notebook notebooks/citation_walk_demo.ipynb")
    logger.info("\n  3. Read the documentation:")
    logger.info("     MULTILINGUAL_QUANTUM_README.md")
    logger.info("\n  4. Run tests:")
    logger.info("     pytest tests/")
    
    logger.info("\n💡 Tips:")
    logger.info("  - Set quantum_enabled=False for faster classical-only mode")
    logger.info("  - Use fallback_mode='auto' for automatic error handling")
    logger.info("  - Check logs/ directory for detailed execution logs")
    
    logger.info("\n" + "=" * 70)


def main():
    """Main setup routine"""
    logger.info("=" * 70)
    logger.info("MULTILINGUAL QUANTUM RESEARCH AGENT - SETUP")
    logger.info("=" * 70)
    logger.info("")
    
    # Check Python version
    if not check_python_version():
        logger.error("Setup failed: Python version requirement not met")
        return False
    
    # Install requirements
    if not install_requirements():
        logger.error("Setup failed: Could not install dependencies")
        return False
    
    # Download spaCy models
    download_spacy_models()  # Optional, don't fail on error
    
    # Create directories
    create_directories()
    
    # Check quantum availability
    quantum_ok = check_quantum_availability()
    if not quantum_ok:
        logger.warning("Quantum libraries not fully available - will use classical fallback")
    
    # Check NLP availability
    nlp_ok = check_nlp_availability()
    if not nlp_ok:
        logger.warning("NLP libraries not fully available - some features may be limited")
    
    # Run basic test
    if not run_basic_test():
        logger.error("Setup failed: Basic functionality test failed")
        return False
    
    # Print next steps
    print_next_steps()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
