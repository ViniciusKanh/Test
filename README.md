# MARCA: Modular Association Rule for Classification Algorithms

## Overview

MARCA is a toolkit for classification based on association rules. It provides a modular framework that allows users to choose the best algorithms for each step of the classification process. By leveraging the power of association rules, MARCA enables efficient and interpretable classification models.

## Features

- **Modular Design**: Choose different algorithms for rule generation, pruning, and classification
- **Flexible Rule Generation**: Support for various association rule mining algorithms
- **Customizable Rule Selection**: Multiple strategies for selecting the most relevant rules
- **Interpretable Results**: Classification models based on readable and understandable rules
- **Extensible Framework**: Easy integration of new algorithms and techniques

## Installation

### From PyPI

```bash
# Install the latest release
pip install marca
```

### For Development

```bash
# Clone the repository
git clone https://github.com/yourusername/marca.git
cd marca

# Install in development mode
pip install -e .
```

## Usage

```python
from marca import MARCA

# Initialize the classifier with desired components
classifier = MARCA(
    extraction='apriori',
    ranking='chi_square',
    pruning='chi_square',
    prediction='weighted_voting'
)

# Train the classifier
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)
```

## Documentation

Detailed documentation is available at [docs/](docs/). This includes:

- API reference
- Algorithm descriptions
- Performance benchmarks
- Usage examples

## Project Structure

```
marca/
├── core/            # Core framework components
├── preprocessing/   # Data preprocessing algorithms
├── extraction/      # Association rule mining algorithms
├── ranking/         # Ranking algorithms
├── pruning/         # Rule pruning and selection
├── prediction/      # Rule prediction step
├── classifiers/     # Traditional classifiers already implemented
├── utils/           # Utility functions and helpers
examples/            # Example notebooks and scripts
├── tests/               # Test files
├── LICENSE              # License file
├── setup.py             # Setup file
├── .gitignore           # Git ignore file
├── .pre-commit-config.yaml  # Pre-commit configuration
├── .github/             # GitHub configuration
├── docs/                # Documentation
└── README.md            # This file
```

## Contributing

Contributions to MARCA are welcome! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MARCA in your research, please cite:

```
@software{marca_toolkit,
  author = {Maicon Dall'Agnol},
  title = {MARCA: Modular Association Rule Classification Approach},
  year = {2025},
  url = {https://github.com/marcaresearch/marca}
}
```

## Contact

For questions and feedback, please open an issue on GitHub or contact [maicon.dallagnol@unesp.br](mailto:maicon.dallagnol@unesp.br).