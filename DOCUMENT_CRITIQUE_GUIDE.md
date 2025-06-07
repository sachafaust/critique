# Document Critique Mode Guide

## Overview

The Document Critique Mode is a new feature that enables direct AI review of human-authored files without the content generation phase. This mode is perfect for getting expert feedback on existing documents, code, papers, proposals, and reports.

## Key Features

- **Direct Document Input**: Upload any local file for critique
- **Multiple AI Critics**: Get feedback from various AI models and expert personas
- **Single Iteration**: Fast, focused feedback in one round
- **Robust File Handling**: Supports various file types with intelligent encoding detection
- **Cost Efficient**: No content generation costs - critics only
- **Same Output Formats**: Human-readable and JSON output options

## Quick Start

### Basic Usage

```bash
# Critique with vanilla AI models
llm-critique document --file paper.txt --critique-models claude-4-sonnet,gpt-4o

# Critique with expert personas
llm-critique document --file proposal.md --personas steve_jobs,ray_dalio

# Use all available expert personas
llm-critique document --file code.py --personas all --personas-model gpt-4o
```

### Cost Estimation

```bash
# Estimate cost before running
llm-critique document --file document.txt --critique-models gpt-4o,claude-4-sonnet --est-cost
```

## Supported File Types

The Document Critique Mode can handle various file types:

### Text Documents
- `.txt` - Plain text files
- `.md` - Markdown documents
- `.rtf` - Rich text format

### Code Files
- `.py` - Python code
- `.js` - JavaScript code
- `.html` - HTML documents
- `.css` - CSS stylesheets
- `.json` - JSON data
- `.yaml`, `.yml` - YAML configuration
- `.xml` - XML documents
- `.sql` - SQL scripts
- `.sh` - Shell scripts

### Data Files
- `.csv` - CSV data files

### Document Files
- `.pdf` - PDF documents (text extraction)
- `.doc`, `.docx` - Word documents (text extraction)

### File Constraints
- **Maximum file size**: 50MB
- **Encoding support**: UTF-8, UTF-8 with BOM, Latin1, CP1252, ASCII
- **Binary file handling**: Extracts readable text from binary files

## Command Options

### Required Options
- `--file PATH` - Path to the document file to critique

### Critique Configuration (Choose One)
- `--critique-models MODELS` - Comma-separated list of AI models
- `--personas PERSONAS` - Comma-separated list of expert personas or "all"

### Optional Configuration
- `--personas-model MODEL` - Global model override when using personas
- `--format FORMAT` - Output format: `human` (default) or `json`
- `--listen FILE` - Save conversation to file for replay
- `--est-cost` - Estimate cost without running

### Global Options
- `--debug` - Enable debug logging
- `--config PATH` - Custom config file path

## Usage Examples

### Academic Paper Review
```bash
# Get feedback from academic experts
llm-critique document --file research_paper.txt --personas ray_dalio,steve_jobs

# Use specific models for academic review
llm-critique document --file thesis.md --critique-models claude-4-opus,gpt-4o
```

### Code Review
```bash
# Review Python code
llm-critique document --file main.py --critique-models gpt-4o,claude-4-sonnet

# Get expert perspective on code architecture
llm-critique document --file architecture.py --personas steve_jobs --personas-model o1
```

### Business Document Review
```bash
# Review business proposal
llm-critique document --file proposal.md --personas ray_dalio,steve_jobs

# Get comprehensive feedback from all experts
llm-critique document --file business_plan.txt --personas all --personas-model claude-4-sonnet
```

### Technical Documentation
```bash
# Review API documentation
llm-critique document --file api_docs.md --critique-models gemini-2.0-flash,gpt-4o

# Get expert feedback on technical writing
llm-critique document --file technical_spec.txt --personas steve_jobs
```

## Output Formats

### Human-Readable Output (Default)
The human-readable format provides:
- Document summary with quality scores
- Expert insights and consensus areas
- Individual critique breakdowns
- Actionable recommendations
- Performance metrics (duration, cost)

### JSON Output
```bash
llm-critique document --file document.txt --critique-models gpt-4o --format json
```

JSON output includes:
- Structured critique data
- Quality metrics and scores
- Persona analysis
- Performance statistics
- Machine-readable recommendations

## Expert Personas vs Vanilla Models

### Expert Personas
Expert personas provide rich, personality-driven feedback with domain expertise:
- **steve_jobs**: Product design, user experience, innovation
- **ray_dalio**: Strategic thinking, principles, systematic analysis
- **elon_musk**: Innovation, engineering, ambitious thinking

```bash
# Use expert personas for rich feedback
llm-critique document --file product_spec.md --personas steve_jobs,ray_dalio
```

### Vanilla Models
Vanilla models provide general-purpose AI feedback without personality:
- **gpt-4o**: Balanced, comprehensive analysis
- **claude-4-sonnet**: Detailed, thoughtful critique
- **gemini-2.0-flash**: Fast, efficient feedback

```bash
# Use vanilla models for general feedback
llm-critique document --file document.txt --critique-models gpt-4o,claude-4-sonnet
```

## Cost Optimization

### Estimate Before Running
Always estimate costs for large documents or multiple critics:
```bash
llm-critique document --file large_document.txt --personas all --personas-model gpt-4o --est-cost
```

### Choose Models Wisely
- **Cost-effective**: `gpt-4o-mini`, `claude-3.5-haiku`, `gemini-1.5-flash-8b`
- **Balanced**: `gpt-4o`, `claude-4-sonnet`, `gemini-2.0-flash`
- **Premium**: `claude-4-opus`, `o1`, `gemini-2.5-pro`

### Single vs Multiple Critics
- **Single critic**: Fast, cost-effective feedback
- **Multiple critics**: Comprehensive, diverse perspectives
- **All personas**: Maximum insight, higher cost

## Error Handling

### File Issues
- **File not found**: Check file path and permissions
- **File too large**: Files over 50MB are rejected
- **Encoding issues**: Tool automatically detects and handles various encodings
- **Empty files**: Files with no readable content are rejected

### Model Issues
- **Missing API keys**: Set required environment variables
- **Model unavailable**: Check API key configuration and model availability
- **Rate limits**: Tool handles rate limiting automatically

### Configuration Issues
- **Invalid personas**: Use `--list-personas` to see available options
- **Invalid models**: Use `--list-models` to see available models
- **Conflicting options**: Cannot use both `--personas` and `--critique-models`

## Integration with Existing Workflow

### Backward Compatibility
The existing critique workflow remains unchanged:
```bash
# Old format still works (with compatibility notice)
llm-critique "Your prompt" --creator-persona steve_jobs --personas ray_dalio

# New explicit format (recommended)
llm-critique critique "Your prompt" --creator-persona steve_jobs --personas ray_dalio
```

### Conversation Recording
Save critiques for later analysis:
```bash
llm-critique document --file document.txt --critique-models gpt-4o --listen critique_session.json
```

### Configuration Files
Use config files for consistent settings:
```yaml
# config.yaml
default_models:
  - gpt-4o
  - claude-4-sonnet
confidence_threshold: 0.8
```

## Best Practices

### Document Preparation
1. **Clean formatting**: Remove unnecessary formatting for better analysis
2. **Clear structure**: Use headings and sections for better critique
3. **Complete content**: Include all relevant context in the document
4. **Reasonable length**: Very long documents may hit token limits

### Critic Selection
1. **Match expertise**: Choose personas relevant to your document type
2. **Diverse perspectives**: Use multiple critics for comprehensive feedback
3. **Cost consideration**: Balance insight quality with cost constraints
4. **Model capabilities**: Consider each model's strengths for your use case

### Output Processing
1. **Review all feedback**: Each critic provides unique insights
2. **Look for consensus**: Areas where multiple critics agree need attention
3. **Prioritize recommendations**: Focus on high-confidence suggestions
4. **Iterate**: Use feedback to improve and re-critique if needed

## Troubleshooting

### Common Issues

**"No API keys found"**
```bash
# Set required environment variables
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

**"File too large"**
- Split large documents into smaller sections
- Use text extraction tools for complex formats
- Consider summarizing very long documents first

**"No critique models specified"**
```bash
# Either specify models directly
llm-critique document --file doc.txt --critique-models gpt-4o

# Or use personas
llm-critique document --file doc.txt --personas steve_jobs

# Or configure defaults in config.yaml
```

**"Encoding detection failed"**
- Save file as UTF-8 encoding
- Check for special characters or binary content
- Try converting file format (e.g., PDF to text)

### Getting Help

```bash
# General help
llm-critique --help

# Document command help
llm-critique document --help

# List available models
llm-critique --list-models

# List available personas
llm-critique --list-personas

# Get persona details
llm-critique --persona-info steve_jobs
```

## Advanced Usage

### Custom Workflows
Combine document critique with other tools:
```bash
# Extract text from PDF, then critique
pdftotext document.pdf - | llm-critique document --file - --critique-models gpt-4o

# Critique multiple files in sequence
for file in *.md; do
  llm-critique document --file "$file" --personas steve_jobs --format json > "${file%.md}_critique.json"
done
```

### Automation
Use in CI/CD pipelines:
```bash
# Automated documentation review
llm-critique document --file README.md --critique-models gpt-4o-mini --format json --est-cost
```

### Integration with IDEs
Many IDEs can integrate with command-line tools for document review workflows.

## Comparison: Document vs Critique Modes

| Feature | Document Mode | Critique Mode |
|---------|---------------|---------------|
| **Purpose** | Review existing documents | Create and iterate content |
| **Input** | Local files | Text prompts |
| **Iterations** | Single round | Multiple rounds |
| **Cost** | Critics only | Creator + Critics |
| **Speed** | Fast | Slower (iterations) |
| **Use Case** | Review, feedback | Content creation |

Choose **Document Mode** when:
- You have existing content to review
- You want fast, focused feedback
- You need cost-effective critique
- You prefer manual revision control

Choose **Critique Mode** when:
- You need content generated from scratch
- You want iterative improvement
- You need AI-driven content creation
- You want automated convergence

## Future Enhancements

Planned improvements for Document Critique Mode:
- **Batch processing**: Critique multiple files at once
- **Template critiques**: Predefined critique templates for specific document types
- **Diff generation**: Automatic suggestion of specific text changes
- **Integration APIs**: REST API for programmatic access
- **Advanced file support**: Better handling of complex document formats
- **Collaborative features**: Multi-user critique sessions

---

For more information, see the main [README.md](README.md) and [USAGE_GUIDE.md](USAGE_GUIDE.md). 