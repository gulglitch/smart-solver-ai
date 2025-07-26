# Vectorly - AI-Powered Multivariable Calculus Problem Solver

## Overview

Vectorly is an intelligent web application designed to solve various types of multivariable calculus problems. Built as part of a Multivariable Calculus course project, it provides automated solutions for line integrals, flux, flow, circulation, work integrals, conservativity checks, and potential functions.

## Features

- **Multiple Integral Types**: Supports line integrals, flux, flow, circulation, work integrals, conservativity checks, and potential functions
- **Symbolic Computation**: Handles complex mathematical expressions using SymPy
- **Smart Problem Matching**: Uses similarity algorithms to find the most relevant problems in the dataset
- **Web Interface**: User-friendly Flask web application
- **Educational Focus**: Provides both computed results and exact solutions for learning

## Problem Types Supported

1. **Line Integrals**: Compute line integrals along parametric curves
2. **Flux**: Calculate flux across various surfaces (circles, triangles, etc.)
3. **Flow**: Determine flow along different paths and curves
4. **Circulation**: Compute circulation around closed curves
5. **Work Integrals**: Calculate work done by force fields
6. **Conservativity Checks**: Determine if vector fields are conservative
7. **Potential Functions**: Find potential functions for conservative fields

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/gulglitch/smart-solver-ai.git
cd vectorly
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   - **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Application
1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Select the type of problem you want to solve and enter your problem statement.

### Using the Command Line Interface
For direct problem solving without the web interface:
```bash
python vectorly.py
```

This will open an interactive menu where you can select different problem types and solve them directly.

## Project Structure

```
vectorly/
├── app.py                 # Flask web application
├── main.py               # Command line interface
├── vectorly.py           # Core mathematical solvers
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── templates/           # HTML templates for web interface
├── static/              # CSS and static files
└── *.json              # Problem datasets for different types
```

## Mathematical Solvers

The project includes specialized classes for each problem type:

- `IntegralSolver`: Base class for line integral calculations
- `FluxCalculator`: Handles flux calculations across various surfaces
- `FlowCalculator`: Computes flow along different paths
- `CirculationCalculator`: Calculates circulation around closed curves
- `SolveWorkIntegral`: Solves work integral problems
- `ConservativeChecker`: Determines if vector fields are conservative
- `PotentialFunctionCalculator`: Finds potential functions for conservative fields

## Dataset

The project includes comprehensive datasets with 100+ problems across different categories:
- `lineintegrals.json`: Line integral problems
- `flux.json`: Flux calculation problems
- `flow.json`: Flow problems
- `circulation.json`: Circulation problems
- `workintegrals.json`: Work integral problems
- `conservativity.json`: Conservativity check problems
- `potentialfunctions.json`: Potential function problems

## Technical Details

### Dependencies
- **Flask**: Web framework for the application
- **SymPy**: Symbolic mathematics library
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing (for integration)

### Key Algorithms
- **Similarity Matching**: Uses SequenceMatcher for finding similar problems
- **Symbolic Integration**: Leverages SymPy for exact mathematical solutions
- **Expression Parsing**: Custom parser for mathematical expressions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built as part of a Multivariable Calculus course project
- Inspired by the need to make complex mathematical concepts more accessible
- Uses SymPy for robust symbolic mathematics

## Contact

For questions or contributions, please open an issue on GitHub or contact the maintainer.

---

**Note**: This project is designed for educational purposes and should be used as a learning tool rather than a replacement for understanding mathematical concepts. 