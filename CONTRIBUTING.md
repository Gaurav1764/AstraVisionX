# Contributing to AstraVisionX

Thank you for your interest in contributing to AstraVisionX! This document provides guidelines for contributing.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/yourusername/AstraVisionX.git
   cd AstraVisionX
   ```
3. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```
4. **Configure** your environment:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests:
   ```bash
   pytest tests/ -v
   ```
4. Commit with a descriptive message:
   ```bash
   git commit -m "Add: description of your change"
   ```
5. Push and open a Pull Request

## Code Style

- Follow PEP 8 conventions
- Use type hints where possible
- Add docstrings to all public functions and classes
- Keep functions focused and reasonably sized

## Project Structure

- **`astravisionx/`** — Core package (models, XAI, chat, utils)
- **`static/`** — Frontend (HTML/CSS/JS)
- **`tests/`** — Test suite
- **`docs/`** — Documentation and research paper

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include steps to reproduce, expected behavior, and actual behavior
- Include Python version, OS, and relevant configuration

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

