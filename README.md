# LLM Critique

A powerful CLI tool that uses a **creator-critic iterative workflow** to generate high-quality content. Multiple AI models work together: one creates content, while others provide detailed critique and feedback, iterating until convergence.

## 🚀 Features

- **Creator-Critic Architecture**: One model creates, others critique and improve iteratively
- **Latest AI Models Support**: OpenAI GPT-4o/o1/o3, Anthropic Claude 4, Google Gemini 2.x
- **Intelligent Convergence**: Automatically stops when quality thresholds are met
- **Cost Estimation**: Pre-execution cost estimates and real-time cost tracking
- **Rich Visual Output**: Beautiful console output with progress tracking and model identification
- **Reasoning Model Support**: Compatible with OpenAI o1/o3 reasoning models
- **Flexible Configuration**: YAML config and environment variables
- **Multiple Output Formats**: Human-readable or JSON output

## 📚 Documentation

### Quick Links
- **[🚀 Getting Started Guide](GETTING_STARTED.md)**: Complete step-by-step setup and usage tutorial
- **[📋 Command Reference](#-command-line-reference)**: Full CLI options and examples
- **[🤖 Model Support](#-supported-models)**: Complete list of supported AI models
- **[⚙️ Configuration](#️-configuration)**: Environment and config file setup

### File Documentation
- **`env.example`**: Template for environment variables with detailed comments
- **`config.yaml`**: Default configuration file with all options
- **`GETTING_STARTED.md`**: Beginner-friendly tutorial with examples and troubleshooting

## 📦 Installation

### Prerequisites

- Python 3.9+
- API keys for the AI providers you want to use

### Quick Start

1. **Clone and Install**:
```bash
git clone https://github.com/yourusername/llm-critique.git
cd llm-critique
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

2. **Set up API Keys** (create `.env` file):
```env
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here  
GOOGLE_API_KEY=your-google-key
```

3. **Run your first critique**:
```bash
python -m llm_critique.main "Write a haiku about coding"
```

## 🎯 Quick Examples

### Basic Usage
```bash
# Simple prompt with default models
python -m llm_critique.main "Explain quantum computing"

# Use specific creator and critic models
python -m llm_critique.main "Write a poem" --creator-model gpt-4o --critique-models claude-4-sonnet,gemini-2.0-flash

# Multiple iterations for complex tasks
python -m llm_critique.main "Design a REST API" --iterations 3

# Read prompt from file
python -m llm_critique.main --file prompt.txt --iterations 2
```

### Cost Estimation
```bash
# Estimate cost before running
python -m llm_critique.main --est-cost "Write a detailed analysis of climate change" --iterations 3

# Estimate cost from file
python -m llm_critique.main --est-cost --file large_prompt.txt --creator-model gpt-4o --critique-models claude-4-opus,gpt-4o-mini
```

### Model Management
```bash
# List all available models
python -m llm_critique.main --list-models

# Use reasoning models
python -m llm_critique.main "Solve this logic puzzle" --creator-model o1-mini --critique-models o3-mini,claude-4-sonnet
```

## 🤖 Supported Models

### OpenAI
- **GPT-4o Series**: `gpt-4o`, `gpt-4o-mini` (multimodal, fast)
- **Reasoning Models**: `o1`, `o1-mini`, `o3`, `o3-mini` (advanced reasoning)
- **Legacy**: `gpt-4`, `gpt-3.5-turbo`

### Anthropic  
- **Claude 4 Series**: `claude-4-opus`, `claude-4-sonnet` (latest generation)
- **Claude 3.x**: `claude-3.7-sonnet`, `claude-3.5-sonnet`, `claude-3.5-haiku`
- **Legacy**: `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`

### Google
- **Gemini 2.x**: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`
- **Legacy**: `gemini-pro`

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in your project root:
```env
# Required: At least one API key
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key  
GOOGLE_API_KEY=your-google-key

# Optional: Customize default behavior
LLM_CRITIQUE_DEFAULT_CREATOR=gpt-4o
LLM_CRITIQUE_DEFAULT_MODELS=claude-4-sonnet,gemini-2.0-flash
LLM_CRITIQUE_MAX_ITERATIONS=2
```

### Configuration File (optional)

Create a `config.yaml` file:
```yaml
# Default models for creator-critic workflow
default_creator: gpt-4o
default_models:
  - claude-4-sonnet
  - gemini-2.0-flash
  - gpt-4o-mini

# Workflow settings
max_iterations: 3
confidence_threshold: 0.85

# Cost and performance
timeout: 120
enable_cost_tracking: true

# Output preferences
output_format: human
debug: false
```

## 📖 How It Works

### Creator-Critic Workflow

1. **Creator Phase**: A designated "creator" model generates initial content
2. **Critic Phase**: Multiple "critic" models analyze and provide structured feedback
3. **Iteration**: Creator improves content based on critic feedback
4. **Convergence**: Process stops when critics are satisfied or max iterations reached

### Example Workflow
```
Iteration 1:
├── Creator (gpt-4o): Generates initial content
├── Critic 1 (claude-4-sonnet): "Good start, but needs more examples" 
├── Critic 2 (gemini-2.0-flash): "Structure is unclear, suggest reorganizing"
└── Decision: Continue (quality score < 85%)

Iteration 2:  
├── Creator (gpt-4o): Improves content based on feedback
├── Critic 1 (claude-4-sonnet): "Much better, examples are clear"
├── Critic 2 (gemini-2.0-flash): "Well structured, minor style tweaks"
└── Decision: Stop (quality score >= 85%)
```

## 🎨 Output Examples

### Human-Readable Format
```
🔄 CREATOR-CRITIC ITERATION RESULTS
================================================================================
📊 Total Iterations: 2
🎯 Convergence Achieved: ✅ Yes  
⚙️  Creator Model: gpt-4o
🔍 Critic Models: claude-4-sonnet, gemini-2.0-flash

==================== ITERATION 1 ====================
🎨 CREATOR OUTPUT (gpt-4o)
Confidence: 80.0%
[Generated content in styled panel]

🔍 CRITICS FEEDBACK (Iteration 1)
  🤖 claude-4-sonnet (Iteration 1)
     📊 Quality Score: 75.0%
     💪 Strengths: Clear writing, good structure
     🔧 Improvements: Add more examples, improve conclusion
     🎯 Decision: 🔄 Continue

🔄 ITERATION SUMMARY
  📝 Requested Iterations: 3
  ✅ Used Iterations: 2  
  🎯 Status: Convergence achieved after 2 iterations
  💡 Early Stop: Stopped 1 iteration early due to quality convergence

⚡ PERFORMANCE
  ⏱️  Total Duration: 12.3s
  💰 Estimated Cost: $0.0087
```

### JSON Format
```bash
python -m llm_critique.main "Your prompt" --format json
```
```json
{
  "execution_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-12-19T10:30:00Z",
  "input": {
    "prompt": "Your prompt",
    "creator_model": "gpt-4o", 
    "critic_models": ["claude-4-sonnet", "gemini-2.0-flash"],
    "max_iterations": 3
  },
  "results": {
    "final_answer": "The final improved content",
    "confidence_score": 0.92,
    "consensus_score": 0.87,
    "total_iterations": 2,
    "convergence_achieved": true
  },
  "performance": {
    "total_duration_ms": 12300,
    "estimated_cost_usd": 0.0087
  }
}
```

## 💰 Cost Estimation

### Pre-Execution Estimation
```bash
# Get cost estimate without running
python -m llm_critique.main --est-cost "Write a research paper on AI ethics" \
  --creator-model gpt-4o \
  --critique-models claude-4-opus,gemini-2.5-pro \
  --iterations 3
```

**Output:**
```
💰 Cost Estimation for LLM Critique Workflow
┌─────────────────┬─────────────┬──────────────────┬──────────────┬────────────┐
│ Component       │ Model       │ Usage            │ Cost per 1K  │ Total Cost │
├─────────────────┼─────────────┼──────────────────┼──────────────┼────────────┤
│ Creator (Iter 1)│ gpt-4o      │ 1,200 in + 500  │ $0.0025      │ $0.0043    │
│ Critic (Iter 1) │ claude-4-opus│ 1,700 in + 200 │ $0.015       │ $0.0285    │
└─────────────────┴─────────────┴──────────────────┴──────────────┴────────────┘

Estimated Total Cost: $0.0674
```

### Real-Time Cost Tracking
The tool tracks actual costs during execution and displays them in the performance summary.

## 🔧 Command Line Reference

### Main Commands
```bash
# Basic execution
python -m llm_critique.main [PROMPT] [OPTIONS]

# Utility commands  
python -m llm_critique.main --list-models      # Show all models
python -m llm_critique.main --est-cost [ARGS]  # Estimate costs
```

### Options
| Option | Description | Example |
|--------|-------------|---------|
| `--file`, `-f` | Read prompt from file | `-f prompt.txt` |
| `--creator-model` | Model for content creation | `--creator-model gpt-4o` |
| `--critique-models` | Comma-separated critic models | `--critique-models claude-4-sonnet,gemini-2.0-flash` |
| `--iterations` | Maximum iterations | `--iterations 3` |
| `--format` | Output format (human/json) | `--format json` |
| `--debug` | Enable debug logging | `--debug` |
| `--config` | Custom config file | `--config custom.yaml` |
| `--list-models` | List available models | `--list-models` |
| `--est-cost` | Estimate cost only | `--est-cost` |
| `--listen` | Save conversation to file | `--listen conversation.json` |
| `--replay` | Replay saved conversation | `--replay conversation.json` |
| `--version` | Show version and exit | `--version` |

## 🧪 Development

### Setup Development Environment
```bash
# Clone and setup
git clone https://github.com/yourusername/llm-critique.git
cd llm-critique

# Create virtual environment  
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llm_critique --cov-report=html

# Run specific test file
pytest tests/test_main.py

# Run with debug output
pytest -v -s
```

### Code Quality
```bash
# Format code
black llm_critique/ tests/
isort llm_critique/ tests/

# Check style
flake8 llm_critique/ tests/

# Type checking
mypy llm_critique/

# Run all quality checks
pre-commit run --all-files
```

## 🏗️ Project Structure

```
llm-critique/
├── llm_critique/
│   ├── core/
│   │   ├── models.py       # LLM client and model management
│   │   ├── chains.py       # Creator-critic workflow chains  
│   │   └── synthesis.py    # Response synthesis and output
│   └── main.py            # CLI entry point
├── tests/
│   ├── test_models.py     # Model integration tests
│   ├── test_chains.py     # Workflow logic tests
│   ├── test_synthesis.py  # Output formatting tests
│   └── conftest.py        # Test configuration
├── config.yaml           # Default configuration
├── .env.example          # Environment variable template
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
└── pyproject.toml        # Project metadata
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes and add tests
4. **Run** quality checks: `pre-commit run --all-files`
5. **Commit** your changes: `git commit -m 'Add amazing feature'`
6. **Push** to the branch: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/llm-critique/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llm-critique/discussions)
- **Documentation**: This README and inline code documentation

## 🔒 Security & Privacy

**Important Security Information:**

### API Key Security
- ✅ **Never commit API keys** to version control
- ✅ Store keys only in `.env` files (excluded from git)
- ✅ Use environment variables for all credentials
- ✅ Keys are automatically redacted from debug output and logs

### Data Handling
- 🔄 **Conversation files** may contain sensitive prompts - stored with restrictive permissions (600)
- 📝 **Log files** use security filtering to prevent credential leakage
- 🚫 **No data sent to external services** beyond the specified AI APIs

### Best Practices
```bash
# ✅ Good - Use environment variables
export OPENAI_API_KEY="sk-your-key-here"

# ❌ Bad - Never hardcode in scripts
api_key = "sk-your-key-here"  # DON'T DO THIS
```

### File Permissions
The tool automatically sets secure file permissions:
- Log files: `600` (owner read/write only)
- Conversation files: `600` (owner read/write only)
- Config directory: `755` (standard directory permissions)

---

**Made with ❤️ for the AI community** 