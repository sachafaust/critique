import asyncio
import click
import json
import logging
import traceback
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install
import os
import yaml
import uuid

from .config import Config, load_config
from .logging.setup import setup_logging, log_execution
from .core.models import LLMClient
from .core.synthesis import ResponseSynthesizer
from .core.conversation import ConversationManager
from .core.personas import UnifiedPersonaManager, PersonaType

# Install rich traceback handler
install()

console = Console()

def get_available_models() -> List[str]:
    """Get list of available models based on API keys."""
    available = []
    
    if os.getenv("OPENAI_API_KEY"):
        available.extend([
            "gpt-4", "gpt-4o", "gpt-4o-mini",
            "o1", "o1-mini", 
            "o3", "o3-mini",
            "gpt-3.5-turbo"
        ])
    
    if os.getenv("ANTHROPIC_API_KEY"):
        available.extend([
            "claude-4-opus", "claude-4-sonnet",
            "claude-3.7-sonnet", 
            "claude-3.5-sonnet", "claude-3.5-haiku",
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"
        ])
    
    if os.getenv("GOOGLE_API_KEY"):
        available.extend([
            "gemini-2.5-pro", "gemini-2.5-flash",
            "gemini-2.0-flash", "gemini-pro", "gemini-1.0-pro"
        ])
    
    return available

def display_available_models():
    """Display all supported models with their details."""
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel
    
    # Model categories and their details
    models_data = {
        "OpenAI Models": {
            "models": [
                ("gpt-4", "GPT-4", "Most capable GPT-4 model", "$0.03"),
                ("gpt-4o", "GPT-4o", "Optimized for multimodal tasks", "$0.0025"),
                ("gpt-4o-mini", "GPT-4o Mini", "Fast and efficient variant", "$0.00015"),
                ("o1", "o1", "Advanced reasoning model", "$0.015"),
                ("o1-mini", "o1 Mini", "Compact reasoning model", "$0.003"),
                ("o3", "o3", "Latest reasoning model (limited access)", "$0.020"),
                ("o3-mini", "o3 Mini", "Efficient reasoning model (limited access)", "$0.005"),
                ("gpt-3.5-turbo", "GPT-3.5 Turbo", "Fast and cost-effective", "$0.002"),
            ],
            "api_key": "OPENAI_API_KEY"
        },
        "Anthropic Claude Models": {
            "models": [
                ("claude-4-opus", "Claude 4 Opus", "Most capable Claude model", "$0.015"),
                ("claude-4-sonnet", "Claude 4 Sonnet", "High-performance balanced model", "$0.003"),
                ("claude-3.7-sonnet", "Claude 3.7 Sonnet", "Extended thinking capabilities", "$0.003"),
                ("claude-3.5-sonnet", "Claude 3.5 Sonnet", "Strong general performance", "$0.003"),
                ("claude-3.5-haiku", "Claude 3.5 Haiku", "Fast and efficient", "$0.0008"),
                ("claude-3-opus", "Claude 3 Opus", "Powerful legacy model", "$0.015"),
                ("claude-3-sonnet", "Claude 3 Sonnet", "Balanced legacy model", "$0.003"),
                ("claude-3-haiku", "Claude 3 Haiku", "Fast legacy model", "$0.00025"),
            ],
            "api_key": "ANTHROPIC_API_KEY"
        },
        "Google Gemini Models": {
            "models": [
                ("gemini-2.5-pro", "Gemini 2.5 Pro", "Most advanced Gemini (limited access)", "$0.00125"),
                ("gemini-2.5-flash", "Gemini 2.5 Flash", "Fast multimodal model", "$0.0005"),
                ("gemini-2.0-flash", "Gemini 2.0 Flash", "Reliable multimodal performance", "$0.001"),
                ("gemini-pro", "Gemini Pro", "Legacy model (maps to 2.0-flash)", "$0.001"),
            ],
            "api_key": "GOOGLE_API_KEY"
        }
    }
    
    # Get available models
    available_models = get_available_models()
    
    console.print()
    console.print(Panel.fit(
        "[bold blue]🤖 LLM Critique Tool - Supported Models[/bold blue]",
        border_style="blue"
    ))
    console.print()
    
    # Display each category
    for category, data in models_data.items():
        # Check if API key is available
        api_key_available = bool(os.getenv(data["api_key"]))
        
        # Create table for this category
        table = Table(title=f"{category}", show_header=True, header_style="bold magenta")
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Description")
        table.add_column("Cost/1K tokens", style="yellow", justify="right")
        table.add_column("Status", justify="center")
        
        for model_id, name, description, cost in data["models"]:
            # Determine status
            if not api_key_available:
                status = Text("❌ No API Key", style="red")
            elif model_id in available_models:
                status = Text("✅ Available", style="green")
            else:
                status = Text("⚠️ Limited Access", style="yellow")
            
            table.add_row(model_id, name, description, cost, status)
        
        console.print(table)
        console.print()
    
    # Display summary
    total_models = sum(len(data["models"]) for data in models_data.values())
    available_count = len(available_models)
    
    # Determine which API keys are missing
    missing_api_keys = []
    for category, data in models_data.items():
        if not os.getenv(data["api_key"]):
            # Extract the friendly name from the API key
            api_key_name = data["api_key"]
            if api_key_name == "OPENAI_API_KEY":
                missing_api_keys.append("OpenAI")
            elif api_key_name == "ANTHROPIC_API_KEY":
                missing_api_keys.append("Anthropic")
            elif api_key_name == "GOOGLE_API_KEY":
                missing_api_keys.append("Google")
    
    # Build missing API keys message
    if missing_api_keys:
        missing_msg = f"• Missing API keys: {', '.join(missing_api_keys)} ({len(missing_api_keys)} of 3)"
    else:
        missing_msg = "• Missing API keys: None (all providers available!)"
    
    console.print(Panel.fit(
        f"[bold]Summary:[/bold]\n"
        f"• Total supported models: {total_models}\n"
        f"• Currently available: {available_count}\n"
        f"{missing_msg}\n\n"
        f"[dim]💡 Set environment variables OPENAI_API_KEY, ANTHROPIC_API_KEY, and/or GOOGLE_API_KEY to unlock more models[/dim]",
        title="[bold green]Model Availability[/bold green]",
        border_style="green"
    ))
    console.print()

def display_available_personas(config_obj=None):
    """Display all available personas (both expert and vanilla)."""
    from rich.table import Table
    from rich.text import Text
    
    console.print()
    console.print(Panel.fit(
        "[bold blue]🎭 LLM Critique Tool - Available Personas[/bold blue]",
        border_style="blue"
    ))
    console.print()
    
    try:
        # Initialize persona manager
        persona_manager = UnifiedPersonaManager(config_obj=config_obj)
        personas_info = persona_manager.list_available_personas()
        
        # Display expert personas
        if personas_info["expert_personas"]:
            expert_table = Table(title="🧠 Expert Personas", show_header=True, header_style="bold magenta")
            expert_table.add_column("Persona Name", style="cyan", no_wrap=True)
            expert_table.add_column("Description", style="green")
            expert_table.add_column("Expertise", style="yellow")
            expert_table.add_column("Status", justify="center")
            
            for persona_name in personas_info["expert_personas"]:
                try:
                    persona_info = persona_manager.get_persona_info(persona_name)
                    description = persona_info.get("description", "")[:60] + "..." if len(persona_info.get("description", "")) > 60 else persona_info.get("description", "")
                    expertise = ", ".join(persona_info.get("expertise_domains", [])[:2])
                    if len(persona_info.get("expertise_domains", [])) > 2:
                        expertise += ", ..."
                    
                    # Check if preferred model is available  
                    preferred_model = persona_info.get("preferred_model", "")
                    if preferred_model in get_available_models():
                        status = Text("✅ Available", style="green")
                    else:
                        status = Text("⚠️ Model Unavailable", style="yellow")
                    
                    expert_table.add_row(persona_name, description, expertise, status)
                except Exception as e:
                    expert_table.add_row(persona_name, f"Error loading: {e}", "", Text("❌ Error", style="red"))
            
            console.print(expert_table)
            console.print()
        
        # Display vanilla models
        if personas_info["vanilla_models"]:
            vanilla_table = Table(title="🤖 Vanilla Model Personas", show_header=True, header_style="bold magenta")
            vanilla_table.add_column("Model Name", style="cyan", no_wrap=True)
            vanilla_table.add_column("Type", style="green")
            vanilla_table.add_column("Description", style="yellow")
            vanilla_table.add_column("Status", justify="center")
            
            for model_name in personas_info["vanilla_models"]:
                # Determine model type
                if "gpt" in model_name or "o1" in model_name or "o3" in model_name:
                    model_type = "OpenAI"
                elif "claude" in model_name:
                    model_type = "Anthropic"
                elif "gemini" in model_name:
                    model_type = "Google"
                else:
                    model_type = "Unknown"
                
                description = f"AI model providing general critique"
                status = Text("✅ Available", style="green")
                
                vanilla_table.add_row(model_name, model_type, description, status)
            
            console.print(vanilla_table)
            console.print()
        
        # Display summary and usage examples
        console.print(Panel.fit(
            f"[bold]Summary:[/bold]\n"
            f"• Expert personas: {personas_info['total_personas']}\n"
            f"• Vanilla models: {personas_info['total_models']}\n\n"
            f"[bold]Usage Examples:[/bold]\n"
            f"[cyan]Expert personas:[/cyan] --personas ray_dalio,steve_jobs\n"
            f"[cyan]Vanilla models:[/cyan] --critique-models claude-4-sonnet,gpt-4o\n"
            f"[cyan]Mixed mode:[/cyan] Not supported (mutually exclusive)\n\n"
            f"[dim]💡 Use --persona-info NAME for detailed information about a specific persona[/dim]",
            title="[bold green]Persona System[/bold green]",
            border_style="green"
        ))
        console.print()
        
    except Exception as e:
        console.print(f"[red]Error displaying personas: {e}[/red]")

def display_persona_info(persona_name: str, config_obj=None):
    """Display detailed information about a specific persona."""
    from rich.table import Table
    
    try:
        persona_manager = UnifiedPersonaManager(config_obj=config_obj)
        persona_info = persona_manager.get_persona_info(persona_name)
        
        console.print()
        console.print(Panel.fit(
            f"[bold blue]🎭 Persona Information: {persona_info['name']}[/bold blue]",
            border_style="blue"
        ))
        console.print()
        
        # Basic info
        basic_table = Table(title="Basic Information", show_header=False)
        basic_table.add_column("Field", style="cyan", width=20)
        basic_table.add_column("Value", style="green")
        
        basic_table.add_row("Name", persona_info['name'])
        basic_table.add_row("Type", persona_info['type'].title())
        basic_table.add_row("Description", persona_info['description'])
        basic_table.add_row("Preferred Model", persona_info['preferred_model'])
        basic_table.add_row("Temperature", str(persona_info['temperature']))
        basic_table.add_row("Max Tokens", str(persona_info['max_tokens']))
        basic_table.add_row("Context Size", f"~{persona_info['context_size_estimate']} tokens")
        
        console.print(basic_table)
        console.print()
        
        # Expert-specific information
        if persona_info['type'] == 'expert':
            # Core principles
            if persona_info.get('core_principles'):
                console.print("[bold]Core Principles:[/bold]")
                for i, principle in enumerate(persona_info['core_principles'][:5], 1):
                    console.print(f"  {i}. {principle}")
                console.print()
            
            # Key questions
            if persona_info.get('key_questions'):
                console.print("[bold]Key Questions They Ask:[/bold]")
                for i, question in enumerate(persona_info['key_questions'][:5], 1):
                    console.print(f"  {i}. {question}")
                console.print()
            
            # Expertise domains
            if persona_info.get('expertise_domains'):
                console.print("[bold]Expertise Domains:[/bold]")
                domains_text = ", ".join(persona_info['expertise_domains'])
                console.print(f"  {domains_text}")
                console.print()
            
            # Red flags
            if persona_info.get('red_flags'):
                console.print("[bold]Red Flags They Watch For:[/bold]")
                for i, flag in enumerate(persona_info['red_flags'][:3], 1):
                    console.print(f"  {i}. {flag}")
                console.print()
            
            # File path
            if persona_info.get('file_path'):
                console.print(f"[dim]Configuration file: {persona_info['file_path']}[/dim]")
                console.print()
        
        # Usage example
        example_command = f"python -m llm_critique.main 'Your prompt here' --personas {persona_name}"
        if persona_info['type'] == 'vanilla':
            example_command = f"python -m llm_critique.main 'Your prompt here' --critique-models {persona_name}"
        
        console.print(Panel.fit(
            f"[bold]Usage Example:[/bold]\n{example_command}",
            title="[bold green]How to Use[/bold green]",
            border_style="green"
        ))
        console.print()
        
    except Exception as e:
        console.print(f"[red]Error displaying persona info for '{persona_name}': {e}[/red]")

def validate_persona_arguments(personas: Optional[str], critique_models: Optional[str], personas_model: Optional[str] = None) -> bool:
    """Validate that personas and critique-models are mutually exclusive and personas-model usage is correct."""
    if personas and critique_models:
        console.print("[red]❌ Error: --personas and --critique-models are mutually exclusive[/red]")
        console.print()
        console.print("[bold]Choose one approach:[/bold]")
        console.print("[cyan]Expert personas:[/cyan] --personas ray_dalio,steve_jobs --personas-model o1")
        console.print("[cyan]All personas:[/cyan] --personas all --personas-model o1") 
        console.print("[cyan]Vanilla models:[/cyan] --critique-models claude-4-sonnet,gpt-4o")
        console.print()
        console.print("[dim]💡 Use --list-personas to see available options[/dim]")
        return False
    
    # Validate personas-model usage
    if personas_model and not personas:
        console.print("[red]❌ Error: --personas-model can only be used with --personas[/red]")
        console.print()
        console.print("[bold]Correct usage:[/bold]")
        console.print("[cyan]--personas ray_dalio,steve_jobs --personas-model o1[/cyan]")
        console.print("[cyan]--personas all --personas-model o1[/cyan]")
        return False
    
    # Validate that --personas all requires --personas-model
    if personas and personas.strip().lower() == "all" and not personas_model:
        console.print("[red]❌ Error: When using --personas all, you must specify --personas-model[/red]")
        console.print()
        console.print("[bold]Correct usage:[/bold]")
        console.print("[cyan]--personas all --personas-model o1[/cyan]")
        console.print("[cyan]--personas all --personas-model gpt-4o[/cyan]")
        console.print()
        console.print("[dim]💡 Use --list-models to see available models[/dim]")
        return False
    
    return True

def validate_config_models(config_obj, available_models: List[str], debug: bool) -> bool:
    """Validate that config models are available and provide guidance if not."""
    if not config_obj:
        return False
    
    # Check if config has any default models
    if not hasattr(config_obj, 'default_models') or not config_obj.default_models:
        console.print("[yellow]⚠️  Config file exists but has no default_models configured[/yellow]")
        return False
    
    # Check if default models are available
    available_defaults = [m for m in config_obj.default_models if m in available_models]
    unavailable_defaults = [m for m in config_obj.default_models if m not in available_models]
    
    if unavailable_defaults:
        console.print(f"[yellow]⚠️  Config default_models not available: {', '.join(unavailable_defaults)}[/yellow]")
        if debug:
            console.print(f"[dim]Available models: {', '.join(available_models)}[/dim]")
        return False
    
    if not available_defaults:
        console.print("[yellow]⚠️  No default models from config are available[/yellow]")
        return False
    
    # Check creator model if specified
    if hasattr(config_obj, 'default_creator') and config_obj.default_creator != "auto":
        if config_obj.default_creator not in available_models:
            console.print(f"[yellow]⚠️  Config default_creator '{config_obj.default_creator}' not available[/yellow]")
            return False
    
    return True

def show_model_requirements():
    """Show clear guidance on model requirements."""
    console.print()
    console.print(Panel.fit(
        "[bold red]❌ Model Configuration Required[/bold red]\n\n"
        "[bold]You need to specify models in one of these ways:[/bold]\n\n"
        "[cyan]Option 1: CLI Arguments[/cyan]\n"
        "python -m llm_critique.main 'Your prompt' \\\n"
        "  --creator-model gpt-4o-mini \\\n"
        "  --critique-models gemini-pro,claude-3-haiku\n\n"
        "[cyan]Option 2: Config File[/cyan]\n"
        "1. Copy template: [dim]cp config.yaml.example config.yaml[/dim]\n"
        "2. Edit config.yaml with your preferred models\n"
        "3. Run: [dim]python -m llm_critique.main 'Your prompt'[/dim]\n\n"
        "[cyan]Option 3: Check Available Models[/cyan]\n"
        "python -m llm_critique.main --list-models",
        title="[bold red]Setup Required[/bold red]",
        border_style="red"
    ))
    console.print()

def print_debug_info(debug: bool, **kwargs):
    """Print debug information if debug mode is enabled, with security filtering."""
    if not debug:
        return
    
    # Security: Filter out sensitive keys
    SENSITIVE_PATTERNS = [
        'api_key', 'secret', 'token', 'password', 'credential', 
        'key', 'auth', 'bearer', 'oauth'
    ]
    
    for key, value in kwargs.items():
        # Check if key contains sensitive patterns
        key_lower = key.lower()
        if any(pattern in key_lower for pattern in SENSITIVE_PATTERNS):
            console.print(f"[dim]DEBUG {key}: [REDACTED][/dim]")
            continue
            
        # Check if value might be a config object with sensitive data
        if hasattr(value, '__dict__'):
            safe_attrs = {}
            for attr_name in dir(value):
                if not attr_name.startswith('_'):
                    attr_lower = attr_name.lower()
                    if any(pattern in attr_lower for pattern in SENSITIVE_PATTERNS):
                        safe_attrs[attr_name] = "[REDACTED]"
                    else:
                        try:
                            attr_value = getattr(value, attr_name)
                            if not callable(attr_value):
                                safe_attrs[attr_name] = attr_value
                        except:
                            safe_attrs[attr_name] = "[ERROR_ACCESSING]"
            console.print(f"[dim]DEBUG {key}: {safe_attrs}[/dim]")
        else:
            console.print(f"[dim]DEBUG {key}: {value}[/dim]")

@click.command()
@click.argument('prompt', required=False)
@click.option('-f', '--file', help='Read prompt from file')
@click.option('--creator-persona', help='Expert persona or model for content creation (e.g., steve_jobs or gpt-4o)')
@click.option('--creator-model', help='[DEPRECATED] Use --creator-persona instead. Model for content creation (e.g., gpt-4o)', hidden=True)
@click.option('--critique-models', help='Comma-separated critic models (e.g., claude-3-sonnet,gemini-pro)')
@click.option('--personas', help='Comma-separated list of expert personas or "all" for all personas (e.g., ray_dalio,steve_jobs or all)')
@click.option('--personas-model', help='Global model to use for all personas when using --personas (e.g., o1)')
@click.option('--format', type=click.Choice(['human', 'json']), default='human')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--config', help='Custom config file path')
@click.option('--iterations', type=int, default=1, help='Number of creator-critic iterations')
@click.option('--listen', help='Save conversation to file for replay')
@click.option('--replay', help='Replay conversation from file')
@click.option('--list-models', is_flag=True, help='List all supported models and exit')
@click.option('--est-cost', is_flag=True, help='Estimate cost without running (requires prompt and models)')
@click.option('--list-personas', is_flag=True, help='List all available personas and exit')
@click.option('--persona-info', help='Display detailed information about a specific persona')
@click.version_option()
def cli(
    prompt: Optional[str],
    file: Optional[str],
    creator_persona: Optional[str],
    creator_model: Optional[str],
    critique_models: Optional[str],
    personas: Optional[str],
    personas_model: Optional[str],
    format: str,
    debug: bool,
    config: Optional[str],
    iterations: int,
    listen: Optional[str],
    replay: Optional[str],
    list_models: bool,
    est_cost: bool,
    list_personas: bool,
    persona_info: Optional[str]
):
    """Multi-LLM critique and synthesis tool with creator-critic iteration.
    
    REQUIREMENTS:
      Either provide personas/models via CLI arguments:
        --creator-persona PERSONA --personas PERSONA1,PERSONA2
        --creator-persona MODEL --critique-models MODEL1,MODEL2
      
      Or create a config.yaml file with default models:
        See config.yaml.example for template
    
    Use --list-personas to see all expert personas and --list-models for supported AI models.
    """
    
    async def run_async():
        nonlocal prompt, creator_persona, creator_model  # Allow modification of these variables
        
        # Handle backward compatibility for --creator-model
        if creator_model and not creator_persona:
            console.print("[yellow]⚠️  --creator-model is deprecated. Use --creator-persona instead.[/yellow]")
            creator_persona = creator_model
        elif creator_model and creator_persona:
            console.print("[red]❌ Error: Cannot use both --creator-model and --creator-persona. Use --creator-persona only.[/red]")
            return

        # Show help if no meaningful arguments provided
        # Note: iterations has default=1, so we exclude it from this check
        meaningful_args = [prompt, file, creator_persona, critique_models, config, listen, replay, list_models, est_cost, debug, personas, list_personas, persona_info]
        if not any(meaningful_args):
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            ctx.exit()
            return

        # Create unique execution ID
        execution_id = str(uuid.uuid4())[:8]

        # Set up logging
        logger = setup_logging(
            level="DEBUG" if debug else "INFO",
            format_type="json",
            trace_id=execution_id,
            log_dir="./logs"
        )

        try:
            if replay:
                conversation_manager = ConversationManager()
                conversation_manager.replay_conversation(replay)
                return
            
            # Load configuration (needed for persona system and cost estimation)
            config_obj = load_config(config)
            
            # Handle persona-specific commands first
            if list_personas:
                display_available_personas(config_obj)
                return
            
            if persona_info:
                display_persona_info(persona_info, config_obj)
                return
            
            # Validate mutually exclusive arguments
            if not validate_persona_arguments(personas, critique_models, personas_model):
                return
            
            # List available models if requested
            if list_models:
                display_available_models()
                return
            
            # Get available models once
            available_models = get_available_models()
            
            # Check if we have any API keys at all
            if not available_models:
                console.print("[red]❌ No API keys found[/red]")
                console.print("Please set at least one API key in your .env file:")
                console.print("• OPENAI_API_KEY")  
                console.print("• ANTHROPIC_API_KEY")
                console.print("• GOOGLE_API_KEY")
                console.print()
                console.print("See env.example for guidance")
                return
            
            # Initialize persona manager
            persona_manager = UnifiedPersonaManager(config_obj=config_obj)
            
            # Determine if we have explicit CLI arguments (either personas or critique models)
            has_cli_personas = bool(personas)
            has_cli_models = bool(critique_models)
            has_explicit_creator = bool(creator_persona)
            
            # Validate that we have some form of critique specification
            if not has_cli_personas and not has_cli_models:
                # Check if config has valid defaults
                config_is_valid = validate_config_models(config_obj, available_models, debug)
                if not config_is_valid:
                    console.print()
                    console.print(Panel.fit(
                        "[bold red]❌ Critique Configuration Required[/bold red]\n\n"
                        "[bold]You need to specify critics in one of these ways:[/bold]\n\n"
                        "[cyan]Option 1: Expert Personas[/cyan]\n"
                        "python -m llm_critique.main 'Your prompt' \\\n"
                        "  --personas ray_dalio,steve_jobs\n\n"
                        "[cyan]Option 2: Mixed Persona Creation[/cyan]\n"
                        "python -m llm_critique.main 'Your prompt' \\\n"
                        "  --creator-persona steve_jobs --personas ray_dalio,warren_buffett\n\n"
                        "[cyan]Option 3: Vanilla Models[/cyan]\n"
                        "python -m llm_critique.main 'Your prompt' \\\n"
                        "  --creator-persona gpt-4o --critique-models claude-4-sonnet,gpt-4o\n\n"
                        "[cyan]Option 4: Config File[/cyan]\n"
                        "1. Copy template: [dim]cp config.yaml.example config.yaml[/dim]\n"
                        "2. Edit config.yaml with your preferred critics\n"
                        "3. Run: [dim]python -m llm_critique.main 'Your prompt'[/dim]\n\n"
                        "[cyan]Option 5: Check Available Options[/cyan]\n"
                        "python -m llm_critique.main --list-personas",
                        title="[bold red]Setup Required[/bold red]",
                        border_style="red"
                    ))
                    console.print()
                    return
            
            # Estimate cost if requested - handle both persona and vanilla modes
            if est_cost:
                estimate_workflow_cost_with_personas(prompt, file, creator_persona, critique_models, personas, personas_model, iterations, config_obj, debug, persona_manager)
                return

            # Load prompt
            if file and os.path.exists(file):
                with open(file, 'r') as f:
                    prompt = f.read().strip()

            if not prompt:
                console.print("[red]Error: No prompt provided[/red]")
                raise click.Abort()

            # Handle persona-based critiques vs vanilla model critiques
            critics_to_use = []
            
            if has_cli_personas:
                # Handle --personas all case
                if personas.strip().lower() == "all":
                    # Get all available personas (excluding template)
                    all_personas = persona_manager.list_available_personas()
                    requested_personas = [p for p in all_personas["expert_personas"] if p != "persona_template"]
                    console.print(f"[cyan]🎭 Using all {len(requested_personas)} available personas with model {personas_model}[/cyan]")
                    if debug:
                        console.print(f"[dim]Personas: {', '.join(requested_personas)}[/dim]")
                else:
                    # Parse specific personas
                    requested_personas = [p.strip() for p in personas.split(',')]
                
                print_debug_info(debug, requested_personas=requested_personas, personas_model=personas_model)
                
                # Validate personas-model is available if specified
                if personas_model and personas_model not in available_models:
                    console.print(f"[red]Error: Personas model '{personas_model}' not available[/red]")
                    console.print(f"Available models: {', '.join(available_models)}")
                    return
                
                validation_result = persona_manager.validate_persona_combination(requested_personas)
                
                if not validation_result["valid"]:
                    console.print("[red]❌ Persona validation failed:[/red]")
                    for error in validation_result["errors"]:
                        console.print(f"  • {error}")
                    return
                
                # Show warnings if any
                for warning in validation_result["warnings"]:
                    console.print(f"[yellow]⚠️  {warning}[/yellow]")
                
                # Create critics from personas
                for persona_name in requested_personas:
                    try:
                        persona_config = persona_manager.get_persona(persona_name)
                        
                        # Override the preferred model if personas_model is specified
                        if personas_model:
                            # Create a copy of the persona config with the global model
                            import copy
                            persona_config = copy.deepcopy(persona_config)
                            persona_config.preferred_model = personas_model
                        
                        critics_to_use.append({
                            "name": persona_config.name,
                            "type": "persona",
                            "config": persona_config
                        })
                    except Exception as e:
                        console.print(f"[red]Error loading persona '{persona_name}': {e}[/red]")
                        return
                
                if not critics_to_use:
                    console.print("[red]Error: No valid personas could be loaded[/red]")
                    return
            else:
                # Parse vanilla models (backward compatibility)
                if critique_models:
                    requested_models = [m.strip() for m in critique_models.split(',')]
                    print_debug_info(debug, requested_models=requested_models)
                    models_to_use = [m for m in requested_models if m in available_models]
                    
                    # Warn about unavailable models
                    unavailable = [m for m in requested_models if m not in available_models]
                    if unavailable:
                        console.print(f"[yellow]Warning: Models not available: {', '.join(unavailable)}[/yellow]")
                    
                    if not models_to_use:
                        console.print("[red]Error: No critic models available from your selection[/red]")
                        console.print(f"Available models: {', '.join(available_models)}")
                        return
                else:
                    # Use validated config models  
                    models_to_use = [m for m in config_obj.default_models if m in available_models]
                
                # Create vanilla critics from models
                for model_name in models_to_use:
                    try:
                        persona_config = persona_manager.create_vanilla_persona(model_name)
                        critics_to_use.append({
                            "name": model_name,
                            "type": "vanilla",
                            "config": persona_config
                        })
                    except Exception as e:
                        console.print(f"[red]Error creating vanilla persona for '{model_name}': {e}[/red]")
                        return
            
            # Set creator persona - new unified approach
            creator_config = None
            if has_explicit_creator:
                try:
                    # Try to get as persona first (expert or vanilla)
                    creator_config = persona_manager.get_persona(creator_persona)
                    creator_model = creator_config.preferred_model
                    
                    if creator_model not in available_models:
                        console.print(f"[red]Error: Creator persona '{creator_persona}' prefers unavailable model '{creator_model}'[/red]")
                        console.print(f"Available models: {', '.join(available_models)}")
                        return
                        
                    console.print(f"[cyan]🎨 Creator: {creator_config.name} ({creator_config.persona_type.value}) → {creator_model}[/cyan]")
                    
                except ValueError:
                    console.print(f"[red]Error: Unknown creator persona or model '{creator_persona}'[/red]")
                    console.print(f"Available expert personas: {list(persona_manager.get_expert_personas().keys())}")
                    console.print(f"Available models: {', '.join(available_models)}")
                    return
            else:
                # Use validated config creator or auto-select
                creator_model = config_obj.default_creator
                if creator_model == "auto":
                    # Use first critic's preferred model or first available model
                    if critics_to_use:
                        creator_model = critics_to_use[0]["config"].preferred_model
                    else:
                        creator_model = available_models[0]
                elif creator_model not in available_models:
                    console.print(f"[red]Error: Config creator model '{creator_model}' not available[/red]")
                    console.print(f"Available models: {', '.join(available_models)}")
                    return
                
                # Create vanilla persona for the creator model
                try:
                    creator_config = persona_manager.create_vanilla_persona(creator_model)
                except Exception as e:
                    console.print(f"[red]Error creating creator persona for '{creator_model}': {e}[/red]")
                    return
            
            print_debug_info(debug, 
                critics_to_use=[c["name"] for c in critics_to_use],
                creator_persona=creator_config.name,
                creator_model=creator_config.preferred_model,
                creator_type=creator_config.persona_type.value,
                iterations=iterations,
                using_personas=has_cli_personas
            )
            
            # Save conversation start
            conversation_manager = ConversationManager() if listen else None
            if conversation_manager:
                conversation_manager.record_step("conversation_start", {
                    "prompt": prompt,
                    "critics": [c["name"] for c in critics_to_use],
                    "creator": creator_config.name,
                    "creator_type": creator_config.persona_type.value,
                    "mode": "personas" if has_cli_personas else "vanilla"
                })
            
            # Initialize LLM client and synthesizer
            llm_client = LLMClient(config_obj)
            
            # Use persona-aware synthesizer for both creator and critics
            from .core.synthesis import PersonaAwareSynthesizer
            synthesizer = PersonaAwareSynthesizer(
                llm_client=llm_client,
                persona_manager=persona_manager,
                max_iterations=iterations,
                confidence_threshold=config_obj.confidence_threshold
            )
            
            # Execute persona-aware synthesis with creator persona
            results = await synthesizer.synthesize_with_personas(
                prompt=prompt,
                persona_configs=[c["config"] for c in critics_to_use],
                creator_persona=creator_config,  # Pass full creator persona
                output_format=format
            )
            
            # Record model responses
            if conversation_manager:
                conversation_manager.record_step("workflow_complete", {
                    "workflow_results": results["results"]["workflow_results"],
                    "final_answer": results["results"]["final_answer"],
                    "confidence_score": results["results"]["confidence_score"]
                })
            
            # Save conversation
            if listen and conversation_manager:
                conversation_manager.save_conversation(listen)

            # Return JSON output if requested
            if format == "json":
                print(json.dumps(results, indent=2))
            
            # Log successful execution - simplified for compatibility
            if debug:
                console.print(f"[dim]Execution ID: {execution_id}[/dim]")
                console.print(f"[dim]Critics used: {[c['name'] for c in critics_to_use]}[/dim]")
                console.print(f"[dim]Creator: {creator_config.name} ({creator_config.persona_type.value})[/dim]")
                console.print(f"[dim]Mode: {'personas' if has_cli_personas else 'vanilla'}[/dim]")

        except Exception as e:
            if debug:
                console.print(f"[red]Error details: {traceback.format_exc()}[/red]")
            else:
                console.print(f"[red]Error: {str(e)}[/red]")
            
            # Log failed execution with available context
            if debug:
                console.print(f"[dim]Failed execution ID: {execution_id}[/dim]")
                console.print(f"[dim]Error: {str(e)}[/dim]")
            
            raise click.Abort()

    # Run the async function
    asyncio.run(run_async())

def main():
    """Main entry point."""
    cli()

def estimate_tokens(text: str) -> int:
    """Estimate token count from text. Rough approximation: ~4 characters per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)

def estimate_workflow_cost(prompt: Optional[str], file: Optional[str], creator_model: Optional[str], 
                          critique_models: Optional[str], iterations: int, config_obj, debug: bool) -> None:
    """Estimate the cost of running the workflow without actually executing it."""
    from rich.table import Table
    from rich.panel import Panel
    
    try:
        # Load prompt from file if specified
        if file and os.path.exists(file):
            with open(file, 'r') as f:
                prompt = f.read().strip()
        
        if not prompt:
            console.print("[red]Error: No prompt provided for cost estimation[/red]")
            console.print("Use either: python -m llm_critique.main --est-cost 'Your prompt here'")
            console.print("Or: python -m llm_critique.main --est-cost --file prompt.txt")
            return
        
        # Get available models
        available_models = get_available_models()
        
        # Check if we have any API keys at all
        if not available_models:
            console.print("[red]❌ No API keys found[/red]")
            console.print("Please set at least one API key in your .env file")
            return
        
        # Determine if we have explicit CLI model arguments
        has_cli_models = bool(creator_model or critique_models)
        
        # If no CLI models, check if config is valid
        if not has_cli_models:
            config_is_valid = validate_config_models(config_obj, available_models, debug)
            if not config_is_valid:
                show_model_requirements()
                return
        
        # Parse critic models
        if critique_models:
            requested_models = [m.strip() for m in critique_models.split(',')]
            models_to_use = [m for m in requested_models if m in available_models]
            
            unavailable = [m for m in requested_models if m not in available_models]
            if unavailable:
                console.print(f"[yellow]Warning: Models not available: {', '.join(unavailable)}[/yellow]")
            
            if not models_to_use:
                console.print("[red]Error: No critic models available from your selection[/red]")
                console.print(f"Available models: {', '.join(available_models)}")
                return
        else:
            models_to_use = [m for m in config_obj.default_models if m in available_models]
        
        # Set creator model
        if creator_model:
            if creator_model not in available_models:
                console.print(f"[red]Error: Creator model '{creator_model}' not available[/red]")
                console.print(f"Available models: {', '.join(available_models)}")
                return
        else:
            creator_model = config_obj.default_creator
            if creator_model == "auto":
                creator_model = models_to_use[0]
            elif creator_model not in available_models:
                console.print(f"[red]Error: Config creator model '{creator_model}' not available[/red]")
                console.print(f"Available models: {', '.join(available_models)}")
                return
        
        if not models_to_use:
            console.print("[red]Error: No critic models available[/red]")
            return
        
        # Initialize LLM client for cost calculation
        from .core.models import LLMClient
        llm_client = LLMClient(config_obj)
        
        # Estimate token counts
        input_tokens = estimate_tokens(prompt)
        
        # Estimate output tokens (rough approximations based on typical responses)
        creator_output_tokens_per_iteration = 500  # Typical creator response
        critic_output_tokens_per_response = 200    # Typical critic feedback
        
        # Calculate total costs
        total_cost = 0.0
        cost_breakdown = []
        
        console.print()
        console.print(Panel.fit(
            "[bold blue]💰 Cost Estimation for LLM Critique Workflow[/bold blue]",
            border_style="blue"
        ))
        console.print()
        
        # Display input details
        info_table = Table(title="Workflow Configuration", show_header=True, header_style="bold magenta")
        info_table.add_column("Parameter", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Input prompt length", f"{len(prompt)} characters")
        info_table.add_row("Estimated input tokens", f"{input_tokens:,}")
        info_table.add_row("Creator model", creator_model)
        info_table.add_row("Critic models", ", ".join(models_to_use))
        info_table.add_row("Iterations", str(iterations))
        
        console.print(info_table)
        console.print()
        
        # Cost breakdown table
        cost_table = Table(title="Cost Breakdown", show_header=True, header_style="bold magenta")
        cost_table.add_column("Component", style="cyan")
        cost_table.add_column("Model", style="green")
        cost_table.add_column("Usage", style="yellow")
        cost_table.add_column("Cost per 1K", style="yellow")
        cost_table.add_column("Total Cost", style="red", justify="right")
        
        # Creator costs (per iteration)
        for iteration in range(1, iterations + 1):
            # Input cost (prompt + previous context gets larger each iteration)
            context_multiplier = 1 + (iteration - 1) * 0.3  # Context grows ~30% each iteration
            iteration_input_tokens = int(input_tokens * context_multiplier)
            
            input_cost = llm_client.estimate_cost(creator_model, iteration_input_tokens)
            output_cost = llm_client.estimate_cost(creator_model, creator_output_tokens_per_iteration)
            iteration_creator_cost = input_cost + output_cost
            
            cost_table.add_row(
                f"Creator (Iteration {iteration})",
                creator_model,
                f"{iteration_input_tokens:,} in + {creator_output_tokens_per_iteration:,} out",
                f"${llm_client.estimate_cost(creator_model, 1000):.4f}",
                f"${iteration_creator_cost:.4f}"
            )
            
            total_cost += iteration_creator_cost
            cost_breakdown.append(f"Creator iter {iteration}: ${iteration_creator_cost:.4f}")
        
        # Critic costs (per iteration, per model)
        for iteration in range(1, iterations + 1):
            for critic_model in models_to_use:
                # Critics analyze creator output + original prompt
                critic_input_tokens = input_tokens + creator_output_tokens_per_iteration
                
                input_cost = llm_client.estimate_cost(critic_model, critic_input_tokens)
                output_cost = llm_client.estimate_cost(critic_model, critic_output_tokens_per_response)
                critic_cost = input_cost + output_cost
                
                cost_table.add_row(
                    f"Critic (Iteration {iteration})",
                    critic_model,
                    f"{critic_input_tokens:,} in + {critic_output_tokens_per_response:,} out",
                    f"${llm_client.estimate_cost(critic_model, 1000):.4f}",
                    f"${critic_cost:.4f}"
                )
                
                total_cost += critic_cost
                cost_breakdown.append(f"Critic {critic_model} iter {iteration}: ${critic_cost:.4f}")
        
        console.print(cost_table)
        console.print()
        
        # Summary
        console.print(Panel.fit(
            f"[bold]Estimated Total Cost: ${total_cost:.4f}[/bold]\n\n"
            f"[dim]Breakdown:[/dim]\n" + 
            "\n".join(f"• {item}" for item in cost_breakdown[:8]) +  # Show first 8 items
            (f"\n• ... and {len(cost_breakdown) - 8} more" if len(cost_breakdown) > 8 else "") +
            f"\n\n[dim]💡 This is an estimate. Actual costs may vary based on:\n"
            f"• Actual response lengths from models\n"
            f"• Model-specific tokenization differences\n"
            f"• API pricing changes[/dim]",
            title="[bold green]Cost Summary[/bold green]",
            border_style="green"
        ))
        console.print()
        
    except Exception as e:
        console.print(f"[red]Error during cost estimation: {str(e)}[/red]")
        if debug:
            import traceback
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

def estimate_workflow_cost_with_personas(
    prompt: Optional[str], 
    file: Optional[str], 
    creator_persona: Optional[str],
    critique_models: Optional[str], 
    personas: Optional[str],
    personas_model: Optional[str],
    iterations: int, 
    config_obj, 
    debug: bool,
    persona_manager
) -> None:
    """Estimate the cost of running the workflow with persona support."""
    from rich.table import Table
    from rich.panel import Panel
    
    try:
        # Load prompt from file if specified
        if file and os.path.exists(file):
            with open(file, 'r') as f:
                prompt = f.read().strip()
        
        if not prompt:
            console.print("[red]Error: No prompt provided for cost estimation[/red]")
            console.print("Use either: python -m llm_critique.main --est-cost 'Your prompt here'")
            console.print("Or: python -m llm_critique.main --est-cost --file prompt.txt")
            return
        
        # Get available models
        available_models = get_available_models()
        
        # Check if we have any API keys at all
        if not available_models:
            console.print("[red]❌ No API keys found[/red]")
            console.print("Please set at least one API key in your .env file")
            return
        
        # Validate arguments (similar to main workflow)
        if not validate_persona_arguments(personas, critique_models, personas_model):
            return
        
        has_cli_personas = bool(personas)
        has_cli_models = bool(critique_models)
        
        # Determine workflow mode and prepare critics
        critics_info = []
        
        if has_cli_personas:
            # Handle --personas all case for cost estimation
            if personas.strip().lower() == "all":
                all_personas = persona_manager.list_available_personas()
                requested_personas = [p for p in all_personas["expert_personas"] if p != "persona_template"]
                console.print(f"[cyan]🎭 Cost estimating for all {len(requested_personas)} personas with model {personas_model}[/cyan]")
            else:
                requested_personas = [p.strip() for p in personas.split(',')]
            
            # Validate personas-model is available if specified
            if personas_model and personas_model not in available_models:
                console.print(f"[red]Error: Personas model '{personas_model}' not available[/red]")
                console.print(f"Available models: {', '.join(available_models)}")
                return
            
            validation_result = persona_manager.validate_persona_combination(requested_personas)
            
            if not validation_result["valid"]:
                console.print("[red]❌ Persona validation failed for cost estimation:[/red]")
                for error in validation_result["errors"]:
                    console.print(f"  • {error}")
                return
            
            # Get persona configurations
            for persona_name in requested_personas:
                try:
                    persona_config = persona_manager.get_persona(persona_name)
                    
                    # Use global personas_model if specified, otherwise use persona's preferred model
                    effective_model = personas_model if personas_model else persona_config.preferred_model
                    
                    critics_info.append({
                        "name": persona_config.name,
                        "type": persona_config.persona_type.value,
                        "model": effective_model,
                        "context_tokens": persona_config.get_prompt_context_size_estimate(),
                        "temperature": persona_config.temperature
                    })
                except Exception as e:
                    console.print(f"[red]Error loading persona '{persona_name}' for cost estimation: {e}[/red]")
                    return
        elif has_cli_models:
            # Parse vanilla models
            requested_models = [m.strip() for m in critique_models.split(',')]
            models_to_use = [m for m in requested_models if m in available_models]
            
            unavailable = [m for m in requested_models if m not in available_models]
            if unavailable:
                console.print(f"[yellow]Warning: Models not available: {', '.join(unavailable)}[/yellow]")
            
            if not models_to_use:
                console.print("[red]Error: No critic models available from your selection[/red]")
                console.print(f"Available models: {', '.join(available_models)}")
                return
            
            # Create vanilla critic info
            for model_name in models_to_use:
                critics_info.append({
                    "name": model_name,
                    "type": "vanilla",
                    "model": model_name,
                    "context_tokens": 50,  # Minimal context for vanilla
                    "temperature": 0.1
                })
        else:
            # Use config defaults
            config_is_valid = validate_config_models(config_obj, available_models, debug)
            if not config_is_valid:
                console.print("[red]Error: No valid default models in configuration for cost estimation[/red]")
                return
            
            models_to_use = [m for m in config_obj.default_models if m in available_models]
            for model_name in models_to_use:
                critics_info.append({
                    "name": model_name,
                    "type": "vanilla",
                    "model": model_name,
                    "context_tokens": 50,
                    "temperature": 0.1
                })
        
        # Determine creator persona
        creator_info = {}
        if creator_persona:
            try:
                creator_config = persona_manager.get_persona(creator_persona)
                if creator_config.preferred_model not in available_models:
                    console.print(f"[red]Error: Creator persona '{creator_persona}' prefers unavailable model '{creator_config.preferred_model}'[/red]")
                    console.print(f"Available models: {', '.join(available_models)}")
                    return
                
                creator_info = {
                    "name": creator_config.name,
                    "type": creator_config.persona_type.value,
                    "model": creator_config.preferred_model,
                    "context_tokens": creator_config.get_prompt_context_size_estimate(),
                    "temperature": creator_config.temperature
                }
                
            except ValueError:
                console.print(f"[red]Error: Unknown creator persona or model '{creator_persona}'[/red]")
                console.print(f"Available expert personas: {list(persona_manager.get_expert_personas().keys())}")
                console.print(f"Available models: {', '.join(available_models)}")
                return
        else:
            creator_model = config_obj.default_creator
            if creator_model == "auto":
                creator_model = critics_info[0]["model"] if critics_info else available_models[0]
            elif creator_model not in available_models:
                console.print(f"[red]Error: Config creator model '{creator_model}' not available[/red]")
                console.print(f"Available models: {', '.join(available_models)}")
                return
            
            # Create vanilla creator info
            creator_info = {
                "name": creator_model,
                "type": "vanilla",
                "model": creator_model,
                "context_tokens": 50,
                "temperature": 0.1
            }
        
        # Initialize LLM client for cost calculation
        from .core.models import LLMClient
        llm_client = LLMClient(config_obj)
        
        # Estimate token counts
        input_tokens = estimate_tokens(prompt)
        
        # Estimate output tokens (based on workflow type and creator persona type)
        creator_output_tokens_per_iteration = 800 if creator_info["type"] == "expert" else 500  # Expert creators may be more verbose
        critic_output_tokens_per_response = 300 if has_cli_personas else 200   # Expert personas give more detailed feedback
        
        # Calculate total costs
        total_cost = 0.0
        cost_breakdown = []
        
        console.print()
        console.print(Panel.fit(
            f"[bold blue]💰 Creator Persona + {('Expert Personas' if has_cli_personas else 'Vanilla Models')} Cost Estimation[/bold blue]",
            border_style="blue"
        ))
        console.print()
        
        # Display workflow configuration
        info_table = Table(title="Workflow Configuration", show_header=True, header_style="bold magenta")
        info_table.add_column("Parameter", style="cyan")
        info_table.add_column("Value", style="green")
        
        info_table.add_row("Input prompt length", f"{len(prompt)} characters")
        info_table.add_row("Estimated input tokens", f"{input_tokens:,}")
        
        # Creator display with persona info
        creator_display = f"🎭 {creator_info['name']} ({creator_info['type']}) → {creator_info['model']}" if creator_info['type'] == 'expert' else f"🤖 {creator_info['name']}"
        info_table.add_row("Creator persona", creator_display)
        
        info_table.add_row("Workflow mode", "Expert Personas" if has_cli_personas else "Vanilla Models")
        critics_display = ", ".join([c["name"] for c in critics_info])
        info_table.add_row("Critics", critics_display)
        info_table.add_row("Iterations", str(iterations))
        
        # Context token summary
        total_context_tokens = creator_info["context_tokens"] + sum(c["context_tokens"] for c in critics_info)
        info_table.add_row("Total persona context", f"{total_context_tokens:,} tokens")
        
        console.print(info_table)
        console.print()
        
        # Cost breakdown table
        cost_table = Table(title="Cost Breakdown", show_header=True, header_style="bold magenta")
        cost_table.add_column("Component", style="cyan")
        cost_table.add_column("Persona/Model", style="green")
        cost_table.add_column("Usage", style="yellow")
        cost_table.add_column("Cost per 1K", style="yellow")
        cost_table.add_column("Total Cost", style="red", justify="right")
        
        # Creator costs (per iteration)
        for iteration in range(1, iterations + 1):
            # Input cost (prompt + previous context + creator persona context)
            context_multiplier = 1 + (iteration - 1) * 0.3  # Context grows ~30% each iteration
            base_input_tokens = int(input_tokens * context_multiplier)
            total_input_tokens = base_input_tokens + creator_info["context_tokens"]
            
            input_cost = llm_client.estimate_cost(creator_info["model"], total_input_tokens)
            output_cost = llm_client.estimate_cost(creator_info["model"], creator_output_tokens_per_iteration)
            iteration_creator_cost = input_cost + output_cost
            
            creator_type_display = f"🎭 {creator_info['name']}" if creator_info["type"] == "expert" else f"🤖 {creator_info['name']}"
            
            cost_table.add_row(
                f"Creator (Iteration {iteration})",
                creator_type_display,
                f"{total_input_tokens:,} in + {creator_output_tokens_per_iteration:,} out",
                f"${llm_client.estimate_cost(creator_info["model"], 1000):.4f}",
                f"${iteration_creator_cost:.4f}"
            )
            
            total_cost += iteration_creator_cost
            cost_breakdown.append(f"Creator {creator_info['name']} iter {iteration}: ${iteration_creator_cost:.4f}")
        
        # Critic costs (per iteration, per critic)
        for iteration in range(1, iterations + 1):
            for critic_info in critics_info:
                critic_name = critic_info["name"]
                critic_model = critic_info["model"]
                context_tokens = critic_info["context_tokens"]
                
                # Critics analyze creator output + original prompt + persona context
                base_input_tokens = input_tokens + creator_output_tokens_per_iteration
                total_input_tokens = base_input_tokens + context_tokens
                
                input_cost = llm_client.estimate_cost(critic_model, total_input_tokens)
                output_cost = llm_client.estimate_cost(critic_model, critic_output_tokens_per_response)
                critic_cost = input_cost + output_cost
                
                critic_type_display = f"🧠 {critic_name}" if critic_info["type"] == "expert" else f"🤖 {critic_name}"
                
                cost_table.add_row(
                    f"Critic (Iteration {iteration})",
                    critic_type_display,
                    f"{total_input_tokens:,} in + {critic_output_tokens_per_response:,} out",
                    f"${llm_client.estimate_cost(critic_model, 1000):.4f}",
                    f"${critic_cost:.4f}"
                )
                
                total_cost += critic_cost
                cost_breakdown.append(f"Critic {critic_name} iter {iteration}: ${critic_cost:.4f}")
        
        console.print(cost_table)
        console.print()
        
        # Enhanced summary for creator persona mode
        creator_benefits = ""
        if creator_info["type"] == "expert":
            creator_benefits = f"\n[bold]🎭 Expert Creator Benefits:[/bold]\n" \
                              f"• Authentic voice and perspective from {creator_info['name']}\n" \
                              f"• Domain-specific insights and communication style\n" \
                              f"• Rich context with {creator_info['context_tokens']:,} creator persona tokens\n" \
                              f"• Consistent character throughout iterations\n"
        else:
            creator_benefits = f"\n[bold]🤖 Vanilla Creator Benefits:[/bold]\n" \
                              f"• General-purpose content generation\n" \
                              f"• Low context overhead for cost efficiency\n" \
                              f"• Reliable, professional output\n"
        
        mode_benefits = ""
        if has_cli_personas:
            mode_benefits = f"\n[bold]🧠 Expert Critics Benefits:[/bold]\n" \
                           f"• Specialized domain knowledge and critique perspectives\n" \
                           f"• Advanced consensus analysis and conflict detection\n" \
                           f"• Rich feedback with {sum(c['context_tokens'] for c in critics_info):,} critic context tokens\n"
        else:
            mode_benefits = f"\n[bold]🤖 Vanilla Critics Benefits:[/bold]\n" \
                           f"• Fast, general-purpose AI critique\n" \
                           f"• Multiple model perspectives for balanced feedback\n"
        
        console.print(Panel.fit(
            f"[bold]Estimated Total Cost: ${total_cost:.4f}[/bold]\n\n"
            f"[dim]Top cost components:[/dim]\n" + 
            "\n".join(f"• {item}" for item in cost_breakdown[:6]) +  # Show first 6 items
            (f"\n• ... and {len(cost_breakdown) - 6} more" if len(cost_breakdown) > 6 else "") +
            creator_benefits + mode_benefits +
            f"\n[dim]💡 This is an estimate. Actual costs may vary based on:\n"
            f"• Actual response lengths from models\n"
            f"• Model-specific tokenization differences\n"
            f"• API pricing changes\n"
            f"• Convergence achieved before max iterations[/dim]",
            title="[bold green]Cost Summary[/bold green]",
            border_style="green"
        ))
        console.print()
        
        # Show optimization tips
        console.print(Panel.fit(
            f"[bold]💡 Creator Persona Optimization Tips:[/bold]\n"
            f"• Expert creators: Rich, authentic content but higher token costs\n"
            f"• Vanilla creators: Cost-efficient, reliable general content\n"
            f"• Mix modes: Expert creator + vanilla critics for balanced approach\n"
            f"• Consider creator-critic personality dynamics for best results",
            title="[bold yellow]Optimization Suggestions[/bold yellow]",
            border_style="yellow"
        ))
        console.print()
        
    except Exception as e:
        console.print(f"[red]Error during cost estimation: {str(e)}[/red]")
        if debug:
            import traceback
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")

if __name__ == "__main__":
    main() 