# MARCA: Modular Association Rule for Classification Algorithms

## Overview

MARCA is a toolkit for classification based on association rules. It provides a modular framework that allows users to choose the best algorithms for each step of the classification process. By leveraging the power of association rules, MARCA enables efficient and interpretable classification models.

## Features

- **Modular Design**: Choose different algorithms for rule extraction, pruning, and classification
- **Flexible Rule Generation**: Support for various association rule mining algorithms (like Apriori)
- **Customizable Pipelines**: Multiple strategies for combining different components
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
git clone https://github.com/marcaresearch/marca.git
cd marca

# Install in development mode
pip install -e .
```

## Usage

Basic example of using MARCA with Apriori rule extraction:

```python
import marca

# Extract association rules using Apriori
extr = marca.extract.Apriori(support=0.1, confidence=0, max_len=5, remove_redundant=False)
rules = extr(x_train, y_train)

# Create and configure a modular classifier
clf = marca.ModularClassifier(rules=rules)

# Train the classifier
clf.fit(x_train, y_train)

# Make predictions and evaluate
score = clf.score(x_test, y_test)
predictions = clf.predict(x_test)
```

Using pipelines for experimentation:

```python
from presets.pipelines import load_pipeline

# Load a predefined pipeline configuration
for pipeline in load_pipeline('default').get():
    # Apply pipeline parameters to the classifier
    clf.set_params(**pipeline.get_params())
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))
```

## Command Line Interface

MARCA also provides a command-line interface for running experiments:

```bash
python main.py --dataset balanced --pipeline default --verbose --workers 1
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
├── marca/           # Core package
│   ├── extract/     # Rule extraction algorithms
│   ├── associative_classifier/ # Classifier implementations
│   └── ...          # Other modules
├── presets/         # Pipeline and experiment presets
├── tests/           # Test files
├── examples/        # Example notebooks and scripts
├── LICENSE          # License file
├── setup.py         # Setup file
├── .gitignore       # Git ignore file
├── docs/            # Documentation
└── README.md        # This file
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