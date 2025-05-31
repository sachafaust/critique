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
            "gemini-2.0-flash", "gemini-pro"
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
    
    console.print(Panel.fit(
        f"[bold]Summary:[/bold]\n"
        f"• Total supported models: {total_models}\n"
        f"• Currently available: {available_count}\n"
        f"• Missing API keys: {3 - sum(1 for data in models_data.values() if os.getenv(data['api_key']))}\n\n"
        f"[dim]💡 Set environment variables OPENAI_API_KEY, ANTHROPIC_API_KEY, and/or GOOGLE_API_KEY to unlock more models[/dim]",
        title="[bold green]Model Availability[/bold green]",
        border_style="green"
    ))
    console.print()

def print_debug_info(debug: bool, **kwargs):
    """Print debug information if debug mode is enabled."""
    if debug:
        for key, value in kwargs.items():
            console.print(f"[dim]DEBUG {key}: {value}[/dim]")

@click.command()
@click.argument('prompt', required=False)
@click.option('-f', '--file', help='Read prompt from file')
@click.option('--creator-model', help='Model for content creation (e.g., gpt-4)')
@click.option('--critique-models', help='Comma-separated critic models (e.g., claude-3-sonnet,gemini-pro)')
@click.option('--format', type=click.Choice(['human', 'json']), default='human')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--config', help='Custom config file path')
@click.option('--iterations', type=int, default=1, help='Number of creator-critic iterations')
@click.option('--listen', help='Save conversation to file for replay')
@click.option('--replay', help='Replay conversation from file')
@click.option('--list-models', is_flag=True, help='List all supported models and exit')
@click.option('--est-cost', is_flag=True, help='Estimate cost without running (requires prompt and models)')
@click.version_option()
def cli(
    prompt: Optional[str],
    file: Optional[str],
    creator_model: Optional[str],
    critique_models: Optional[str],
    format: str,
    debug: bool,
    config: Optional[str],
    iterations: int,
    listen: Optional[str],
    replay: Optional[str],
    list_models: bool,
    est_cost: bool
):
    """Multi-LLM critique and synthesis tool with creator-critic iteration.
    
    Use --list-models to see all supported AI models and their availability.
    """
    
    async def run_async():
        nonlocal prompt, creator_model  # Allow modification of these variables
        
        # Show help if no arguments provided
        if not any([prompt, file, creator_model, critique_models, debug, config, iterations, listen, replay, list_models, est_cost]):
            click.echo(cli.get_help())
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
            
            # List available models if requested
            if list_models:
                display_available_models()
                return

            # Load configuration (needed for cost estimation and normal operation)
            config_obj = load_config(config)
            
            # Estimate cost if requested
            if est_cost:
                estimate_workflow_cost(prompt, file, creator_model, critique_models, iterations, config_obj, debug)
                return

            # Load prompt
            if file and os.path.exists(file):
                with open(file, 'r') as f:
                    prompt = f.read().strip()

            if not prompt:
                console.print("[red]Error: No prompt provided[/red]")
                raise click.Abort()

            # Parse model list
            if critique_models:
                requested_models = [m.strip() for m in critique_models.split(',')]
                print_debug_info(debug, requested_models=requested_models)
                # Filter to only available models
                available_models = get_available_models()
                models_to_use = [m for m in requested_models if m in available_models]
                
                # Warn about unavailable models
                unavailable = [m for m in requested_models if m not in available_models]
                if unavailable:
                    console.print(f"[yellow]Warning: Models not available: {', '.join(unavailable)}[/yellow]")
                
                if not models_to_use:
                    console.print("[red]Error: No available models found[/red]")
                    return
            else:
                # Use default models from config
                available_models = get_available_models()
                models_to_use = [m for m in config_obj.default_models if m in available_models]
                if not models_to_use:
                    models_to_use = available_models
            
            # Set creator model
            if not creator_model:
                creator_model = config_obj.default_creator
            if creator_model == "auto":
                creator_model = models_to_use[0] if models_to_use else "gpt-4"
            
            print_debug_info(debug, 
                models_to_use=models_to_use,
                creator_model=creator_model,
                iterations=iterations
            )
            
            # Save conversation start
            conversation_manager = ConversationManager() if listen else None
            if conversation_manager:
                conversation_manager.record_step("conversation_start", {
                    "prompt": prompt,
                    "models": models_to_use,
                    "creator": creator_model
                })
            
            # Initialize LLM client and synthesizer
            llm_client = LLMClient(config_obj)
            synthesizer = ResponseSynthesizer(
                llm_client=llm_client,
                max_iterations=iterations,
                confidence_threshold=config_obj.confidence_threshold
            )
            
            # Execute synthesis
            results = await synthesizer.synthesize(
                prompt=prompt,
                models=models_to_use,
                creator_model=creator_model,
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
                console.print(f"[dim]Models used: {models_to_use}[/dim]")
                console.print(f"[dim]Creator model: {creator_model}[/dim]")

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
        
        # Parse critic models
        if critique_models:
            requested_models = [m.strip() for m in critique_models.split(',')]
            models_to_use = [m for m in requested_models if m in available_models]
        else:
            models_to_use = [m for m in config_obj.default_models if m in available_models]
            if not models_to_use:
                models_to_use = available_models[:2]  # Use first 2 available as fallback
        
        # Set creator model
        if not creator_model:
            creator_model = config_obj.default_creator
        if creator_model == "auto":
            creator_model = available_models[0] if available_models else "gpt-4o"
        
        if creator_model not in available_models:
            console.print(f"[red]Error: Creator model '{creator_model}' not available[/red]")
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

if __name__ == "__main__":
    main() 