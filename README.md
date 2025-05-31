# LLM Critique

A powerful CLI tool for querying multiple LLMs simultaneously, applying critique and revision using LangChain, and outputting high-quality synthesized answers.

## Features

- Query multiple LLMs in parallel (OpenAI, Anthropic, Google)
- Smart critique and analysis of responses
- Iterative synthesis with confidence scoring
- Rich console output with progress tracking
- Comprehensive logging and execution history
- Cost estimation and performance metrics
- Configurable via YAML and environment variables

## Installation

### From Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-critique.git
cd llm-critique
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

### Development Installation

For development, install with additional dependencies:
```bash
pip install -e ".[dev]"
pre-commit install
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:
```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
```

### Configuration File

Create a `config.yaml` file in your project root:
```yaml
default_models:
  - gpt-4
  - claude-3-opus
  - gemini-pro
default_resolver: gpt-4
max_iterations: 3
confidence_threshold: 0.8
timeout: 30
log_dir: logs
```

## Usage

### Basic Usage

```bash
llm-critique "Your prompt here"
```

### Advanced Usage

```bash
# Read prompt from file
llm-critique -f prompt.txt

# Specify models
llm-critique -m gpt-4 -m claude-3-opus "Your prompt"

# Set resolver model
llm-critique -r gpt-4 "Your prompt"

# Output in JSON format
llm-critique -o json "Your prompt"

# Enable debug logging
llm-critique -d "Your prompt"

# Use custom config file
llm-critique -c custom_config.yaml "Your prompt"
```

### Output Format

The tool provides two output formats:

1. Human-readable (default):
```
Model Responses:
- GPT-4: Response 1
- Claude-3: Response 2

Critique Analysis:
- Consistency: 90%
- Completeness: 85%
- Accuracy: 95%

Final Answer:
Synthesized response with explanation

Performance:
- Duration: 2.5s
- Tokens: 250
- Cost: $0.05
```

2. JSON:
```json
{
  "execution_id": "uuid",
  "timestamp": "2024-03-20T12:00:00Z",
  "input": {
    "prompt": "Your prompt",
    "models": ["gpt-4", "claude-3-opus"],
    "resolver_model": "gpt-4"
  },
  "results": {
    "model_responses": [...],
    "critique_analysis": {...},
    "final_answer": {...}
  },
  "performance": {...},
  "quality": {...}
}
```

## Development

### Setup

1. Install development dependencies:
```bash
make install-dev
```

2. Run tests:
```bash
make test
```

3. Check code quality:
```bash
make lint
```

4. Format code:
```bash
make format
```

### Project Structure

```
llm-critique/
├── llm_critique/
│   ├── core/
│   │   ├── models.py      # LLM client implementation
│   │   ├── chains.py      # LangChain implementations
│   │   └── synthesis.py   # Synthesis logic
│   ├── logging/
│   │   └── setup.py       # Logging configuration
│   └── main.py            # CLI entry point
├── tests/
│   ├── test_main.py
│   ├── test_models.py
│   ├── test_chains.py
│   └── test_synthesis.py
├── config.yaml            # Default configuration
├── .env.example          # Example environment variables
└── pyproject.toml        # Project metadata and dependencies
```

### Testing

- Run all tests: `make test`
- Run with coverage: `make test-cov`
- Run specific test file: `pytest tests/test_main.py`
- Run specific test: `pytest tests/test_main.py::test_basic_prompt_execution`

### Code Quality

- Check code style: `make lint`
- Format code: `make format`
- Run pre-commit hooks: `pre-commit run --all-files`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 