# Contributing to P&ID Analyzer

We welcome contributions to the P&ID Analyzer project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Test your changes thoroughly
5. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/pnid-analyzer.git
cd pnid-analyzer

# Create virtual environment
python3 -m venv pnid_env
source pnid_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if available)
pip install -r requirements-dev.txt
```

## Code Style

- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Testing

- Test your changes with different P&ID images
- Verify compatibility with all supported Bedrock models
- Check that the Streamlit interface works correctly
- Ensure error handling works as expected

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all existing tests pass
4. Update the README.md if you change functionality
5. Submit pull request with clear description of changes

## Reporting Issues

When reporting issues, please include:
- Python version
- AWS region
- Bedrock model used
- Error messages (if any)
- Steps to reproduce
- Sample P&ID image (if possible)

## Feature Requests

We welcome feature requests! Please:
- Check existing issues first
- Provide clear use case description
- Explain expected behavior
- Consider implementation complexity

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

Thank you for contributing!
