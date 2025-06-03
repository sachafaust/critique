# 🎭 LLM Critique Tool - Usage Guide

## Quick Start

```bash
# Expert creator + Expert critics (most powerful)
python -m llm_critique.main "Design a mobile app" --creator-persona steve_jobs --personas ray_dalio

# Simple vanilla setup
python -m llm_critique.main "Review my code" --creator-persona gpt-4o --critique-models claude-4-sonnet,grok-beta
```

##  Expert Personas vs 🤖 Vanilla Models

### Expert Personas
- **Rich personality**: Authentic voice, expertise, thinking patterns
- **Examples**: `steve_jobs`, `ray_dalio`, `elon_musk`, `warren_buffett`
- **Features**: Domain expertise, decision frameworks, characteristic language

### Vanilla Models  
- **Standard AI**: Clean, professional responses without personality
- **Examples**: `gpt-4o`, `claude-4-sonnet`, `gemini-2.0-flash`, `grok-beta`, `o3-mini`
- **Features**: Cost-efficient, reliable, general-purpose

## 📋 Complete Usage Patterns

### 1️⃣ Expert Creator + Expert Critics
**Best for**: Rich, authentic content with specialized domain critique

```bash
python -m llm_critique.main "Your prompt" \
  --creator-persona elon_musk \
  --personas steve_jobs,ray_dalio
```

**What happens**: Elon Musk creates content with his characteristic first-principles thinking, while Steve Jobs focuses on user experience and Ray Dalio provides systematic analysis.

### 2️⃣ Expert Creator + Expert Critics (Global Model Override)
**Best for**: Expert personalities but forcing a specific model for cost/speed control

```bash
python -m llm_critique.main "Your prompt" \
  --creator-persona elon_musk \
  --personas steve_jobs,ray_dalio \
  --personas-model o3-mini
```

**What happens**: All expert personas use `o3-mini` instead of their preferred models, maintaining personality but with consistent model behavior.

### 3️⃣ Expert Creator + Vanilla Critics
**Best for**: Authentic creator voice with cost-efficient, general-purpose critique

```bash
python -m llm_critique.main "Your prompt" \
  --creator-persona steve_jobs \
  --critique-models gpt-4o,claude-4-sonnet
```

**What happens**: Steve Jobs creates with his design philosophy, while vanilla models provide professional, unbiased critique.

### 4️⃣ Vanilla Creator + Expert Critics
**Best for**: Professional content creation with specialized expert review

```bash
python -m llm_critique.main "Your prompt" \
  --creator-persona gpt-4o \
  --personas ray_dalio,warren_buffett \
  --personas-model claude-4-sonnet
```

**What happens**: Clean, professional content creation followed by expert business analysis and investment perspective.

### 5️⃣ Vanilla Creator + Vanilla Critics
**Best for**: Cost-efficient, fast, professional workflow

```bash
python -m llm_critique.main "Your prompt" \
  --creator-persona gpt-4o \
  --critique-models claude-4-sonnet,gemini-2.0-flash
```

**What happens**: Standard AI content creation with multi-model professional critique.

### 6️⃣ All Expert Personas (Global Model)
**Best for**: Maximum expertise diversity with model control

```bash
python -m llm_critique.main "Your prompt" \
  --creator-persona elon_musk \
  --personas all \
  --personas-model o3-mini
```

**What happens**: Uses ALL available expert personas as critics, all running on the specified model.

## 🔧 Key Rules & Constraints

### Mutually Exclusive Options
```bash
# ❌ ERROR - Cannot mix these
--personas steve_jobs --critique-models gpt-4o

# ✅ CORRECT - Choose one approach
--personas steve_jobs                    # Expert personas
--critique-models gpt-4o                 # Vanilla models
```

### Global Model Override Rules
```bash
# ✅ WORKS - Overrides all persona models
--personas steve_jobs,ray_dalio --personas-model o3-mini

# ❌ ERROR - personas-model only works with --personas
--critique-models gpt-4o --personas-model o3-mini

# ❌ ERROR - "all" requires global model
--personas all
```

### Creator Options
```bash
# Expert creator (rich personality)
--creator-persona steve_jobs

# Vanilla creator (standard AI)
--creator-persona gpt-4o

# Auto-select (uses first critic's model)
# No --creator-persona flag
```

## 📖 Discovery Commands

```bash
# See all expert personas and vanilla models
python -m llm_critique.main --list-personas

# See all supported AI models and API requirements
python -m llm_critique.main --list-models

# Get detailed info about a specific persona
python -m llm_critique.main --persona-info steve_jobs

# Estimate cost before running
python -m llm_critique.main "Your prompt" --creator-persona steve_jobs --personas ray_dalio --est-cost
```

## 💡 Common Scenarios

### Business Strategy Analysis
```bash
python -m llm_critique.main "Launch strategy for AI product" \
  --creator-persona elon_musk \
  --personas ray_dalio,warren_buffett \
  --personas-model o3-mini
```

### Product Design Review
```bash
python -m llm_critique.main "Mobile app wireframes feedback" \
  --creator-persona steve_jobs \
  --personas simon_sinek
```

### Technical Architecture Review
```bash
python -m llm_critique.main "Microservices architecture proposal" \
  --creator-persona mark_russinovich \
  --critique-models claude-4-sonnet,gemini-2.0-flash
```

### Cost-Efficient General Review
```bash
python -m llm_critique.main "Review my blog post" \
  --creator-persona gpt-4o-mini \
  --critique-models claude-3-haiku,gemini-2.0-flash
```

### Maximum Expert Insight
```bash
python -m llm_critique.main "Evaluate this startup idea" \
  --creator-persona mark_andreessen \
  --personas all \
  --personas-model claude-4-sonnet
```

## ⚠️ Common Mistakes

### ❌ Mixing Persona Types
```bash
# ERROR: Cannot mix --personas and --critique-models
--personas steve_jobs --critique-models gpt-4o
```

### ❌ Wrong Global Model Usage
```bash
# ERROR: --personas-model only works with --personas
--critique-models gpt-4o --personas-model claude-4-sonnet
```

### ❌ Missing Required Model
```bash
# ERROR: --personas all requires --personas-model
--personas all
```

## 🎯 Tips for Best Results

1. **Match creator to content type**: Use Steve Jobs for product design, Elon Musk for technical innovation, Ray Dalio for strategy
2. **Consider cost vs quality**: Expert personas cost more tokens but provide richer insights
3. **Use global model override**: `--personas-model` for consistent behavior across experts
4. **Mix approaches**: Expert creator + vanilla critics for balanced cost/quality
5. **Start with cost estimation**: Use `--est-cost` to understand token usage before running

## 🔧 Advanced Features

### Multiple Iterations
```bash
--iterations 3  # Run 3 creator-critic cycles for refinement
```

### Save Conversations
```bash
--listen my_session.txt  # Save rich conversation log
```

### JSON Output
```bash
--format json  # Machine-readable output
```

### Cost Estimation
```bash
--est-cost  # Estimate tokens and cost without running
```

## 🚀 X AI Grok Models

X AI's Grok models offer real-time access to X (Twitter) data and advanced reasoning capabilities:

### Available Models
- **grok-beta**: Current production model with real-time X data integration
- **grok-3**: Latest flagship model with 1M context window (when available)
- **grok-3-mini**: Faster, cost-efficient version
- **grok-3-reasoning**: Advanced reasoning with "Think" mode
- **grok-2**: Previous generation model

### Setup
```bash
export XAI_API_KEY="your-xai-api-key"
```

### Usage Examples
```bash
# Latest Grok with real-time data
python -m llm_critique.main "Analyze recent tech trends" --creator-persona grok-beta --critique-models grok-3-reasoning

# Cost-efficient setup
python -m llm_critique.main "Quick code review" --creator-persona grok-3-mini --critique-models grok-beta

# Reasoning-focused analysis
python -m llm_critique.main "Complex math problem" --creator-persona grok-3-reasoning --critique-models steve_jobs --personas-model grok-3
```

## 🎯 Common Usage Patterns

---

Need help? Use `python -m llm_critique.main --help` for quick reference or `--list-personas` to explore available experts! 