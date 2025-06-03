# 🚀 Getting Started with LLM Critique

This guide will walk you through setting up and using LLM Critique for the first time.

## 🎯 What is LLM Critique?

LLM Critique uses a **creator-critic workflow** where:
1. A "creator" AI model generates initial content
2. Multiple "critic" AI models provide structured feedback  
3. The creator improves the content based on feedback
4. This process repeats until quality thresholds are met

## 📋 Prerequisites

- **Python 3.9 or higher**
- **At least one AI provider API key**:
  - OpenAI (for GPT-4o, o1, o3 models)
  - Anthropic (for Claude 4 models)  
  - Google (for Gemini 2.x models)
  - X AI (for Grok models with real-time data)

## 🛠️ Step 1: Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/llm-critique.git
cd llm-critique
```

### Create Virtual Environment
```bash
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -e .
```

## 🔑 Step 2: API Key Setup

### Get Your API Keys

#### OpenAI (Recommended)
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Copy the key (starts with `sk-`)

#### Anthropic (Recommended)
1. Go to [Anthropic Console](https://console.anthropic.com/account/keys)
2. Click "Create Key"
3. Copy the key (starts with `sk-ant-`)

#### Google (Optional)
1. Go to [Google Cloud Console](https://console.developers.google.com/apis/credentials)
2. Create a new API key
3. Enable the Generative AI API

#### X AI (Optional)
1. Go to [X AI Console](https://console.x.ai/)
2. Create an API key
3. Copy the key (starts with `xai-`)

### Configure Environment Variables
```bash
# Copy the example file
cp env.example .env

# Edit the .env file with your API keys
nano .env  # or use your preferred editor
```

Add your API keys to `.env`:
```env
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
XAI_API_KEY=xai-your-xai-key-here
```

## 🎮 Step 3: First Run

### Test Your Setup
```bash
python -m llm_critique.main "Write a haiku about programming"
```

**Expected Output:**
```
🔄 CREATOR-CRITIC ITERATION RESULTS
================================================================================
📊 Total Iterations: 1
🎯 Convergence Achieved: ✅ Yes
⚙️  Creator Model: gpt-4o-mini
🔍 Critic Models: claude-3-haiku

==================== ITERATION 1 ====================
🎨 CREATOR OUTPUT (gpt-4o-mini)
[Beautiful haiku appears here]

🔍 CRITICS FEEDBACK (Iteration 1)
  🤖 claude-3-haiku (Iteration 1)
     📊 Quality Score: 85.0%
     💪 Strengths: [Feedback appears here]
```

### Check Available Models
```bash
python -m llm_critique.main --list-models
```

This shows all supported models and their availability based on your API keys.

## 🎯 Step 4: Basic Usage Examples

### Simple Content Creation
```bash
# Short content
python -m llm_critique.main "Explain machine learning in simple terms"

# From a file
echo "Write a product description for a smart watch" > prompt.txt
python -m llm_critique.main --file prompt.txt
```

### Specify Models
```bash
# Choose your creator and critics
python -m llm_critique.main "Design a mobile app" \
  --creator-model gpt-4o \
  --critique-models claude-4-sonnet,gemini-2.0-flash

# Use X AI Grok models
python -m llm_critique.main "Analyze recent tech trends" \
  --creator-model grok-3 \
  --critique-models grok-beta,claude-4-sonnet
```

### Multiple Iterations
```bash
# Let the system iterate up to 3 times for complex tasks
python -m llm_critique.main "Write a business plan for a tech startup" \
  --iterations 3
```

### Cost Estimation
```bash
# Estimate cost before running
python -m llm_critique.main --est-cost "Write a detailed research paper" \
  --creator-model gpt-4o \
  --critique-models claude-4-opus \
  --iterations 3

# Estimate Grok model costs (higher pricing)
python -m llm_critique.main --est-cost "Complex analysis task" \
  --creator-model grok-3 \
  --critique-models grok-3-reasoning,claude-4-sonnet
```

## 🎨 Step 5: Understanding the Output

### Human-Readable Format (Default)
The output shows:
- **Header**: Summary of iterations, convergence, and models used
- **Iteration Details**: For each iteration:
  - Creator output with confidence score
  - Critic feedback with quality scores and suggestions
  - Decision to continue or stop
- **Final Results**: The improved final answer
- **Iteration Summary**: Efficiency metrics
- **Performance**: Duration and cost

### JSON Format
```bash
python -m llm_critique.main "Your prompt" --format json > results.json
```

Perfect for programmatic use or integration with other tools.

## 💡 Step 6: Tips for Best Results

### Choosing Models

**For Creative Content:**
- Creator: `gpt-4o` or `claude-4-sonnet`
- Critics: `claude-4-sonnet`, `gemini-2.0-flash`

**For Technical Content:**
- Creator: `gpt-4o` or `o1-mini` (reasoning)
- Critics: `claude-4-opus`, `gpt-4o-mini`

**For Real-Time Analysis:**
- Creator: `grok-beta` or `grok-3` (X data access)  
- Critics: `grok-3-reasoning`, `claude-4-sonnet`

**For Cost-Effectiveness:**
- Creator: `gpt-4o-mini` or `claude-3.5-haiku`
- Critics: `gemini-2.0-flash`, `claude-3.5-haiku`

### Iteration Guidelines
- **Simple tasks**: 1 iteration
- **Medium complexity**: 2-3 iterations  
- **Complex projects**: 3+ iterations
- **Cost-sensitive**: Use `--est-cost` first

### Prompt Writing Tips
- Be specific about what you want
- Include context and requirements
- Mention the target audience
- Specify format preferences

## ⚙️ Step 7: Advanced Configuration

### Custom Configuration File
```bash
# Create custom config
cp config.yaml my-config.yaml
# Edit my-config.yaml with your preferences
python -m llm_critique.main "Your prompt" --config my-config.yaml
```

### Environment Defaults
Set defaults in your `.env` file:
```env
LLM_CRITIQUE_DEFAULT_CREATOR=gpt-4o
LLM_CRITIQUE_DEFAULT_MODELS=claude-4-sonnet,gemini-2.0-flash
LLM_CRITIQUE_MAX_ITERATIONS=2
```

## 🔍 Troubleshooting

### Common Issues

**"No API key found"**
- Check your `.env` file exists and has correct API keys
- Ensure you're in the right directory
- Verify API key format (OpenAI starts with `sk-`, X AI with `xai-`)

**"Model not available"**
- Run `--list-models` to see available models
- Check your API key has access to the model
- Some models require special access (like o3 models)

**High costs**
- Use `--est-cost` before running
- Choose smaller models for critics (`gpt-4o-mini`, `claude-3.5-haiku`)
- Reduce `--iterations` count
- Note: Grok models have higher per-token costs

**Poor quality results**
- Try different model combinations
- Increase `--iterations` for complex tasks
- Make your prompt more specific
- Use reasoning models (`o1`, `o3`, `grok-3-reasoning`) for complex logic

### Getting Help
```bash
# See all options
python -m llm_critique.main --help

# Debug mode for troubleshooting
python -m llm_critique.main "Your prompt" --debug
```

## 🎉 What's Next?

Now you're ready to use LLM Critique! Try these examples:

```bash
# Creative writing
python -m llm_critique.main "Write a sci-fi short story" --iterations 2

# Technical documentation  
python -m llm_critique.main "Document this API endpoint" --file api_spec.txt

# Business content
python -m llm_critique.main "Create a marketing strategy" \
  --creator-model claude-4-sonnet \
  --critique-models gpt-4o,gemini-2.0-flash \
  --iterations 3

# Real-time analysis with Grok
python -m llm_critique.main "Analyze current tech trends" \
  --creator-model grok-beta \
  --critique-models grok-3-reasoning,claude-4-sonnet

# Code review and improvement
python -m llm_critique.main "Review and improve this Python function" \
  --file code_snippet.py \
  --creator-model o1-mini \
  --critique-models gpt-4o,claude-4-sonnet
```

Happy creating! 🚀 