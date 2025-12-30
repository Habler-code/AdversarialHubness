# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for HubScan."""

import click
import sys
import json
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback to basic console
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

from .config import Config
from .core.scanner import Scanner
from .utils.logging import setup_logger, get_logger
from .core.scoring import Verdict

console = Console() if RICH_AVAILABLE else Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.version_option(version="0.1.0", prog_name="hubscan")
def cli(verbose: bool):
    """HubScan - Adversarial Hubness Detection for RAG Systems.
    
    A security scanner that detects adversarial hubs in FAISS vector indices
    and RAG/retrieval systems.
    """
    if verbose:
        setup_logger("hubscan", level=10)  # DEBUG
    else:
        setup_logger("hubscan", level=20)  # INFO


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--output", "-o", type=str, help="Output directory (overrides config)")
@click.option("--summary-only", is_flag=True, help="Show only summary, don't save full reports")
@click.option("--ranking-method", type=click.Choice(["vector", "hybrid", "lexical"]), help="Ranking method to use")
@click.option("--hybrid-alpha", type=float, default=0.5, help="Weight for vector search in hybrid mode (0.0-1.0)")
@click.option("--query-texts", type=click.Path(exists=True), help="Path to query texts file (for lexical/hybrid search)")
@click.option("--rerank", is_flag=True, help="Enable reranking as post-processing step")
@click.option("--rerank-method", type=str, default="default", help="Reranking method name (default: default)")
@click.option("--rerank-top-n", type=int, default=100, help="Number of candidates to retrieve before reranking")
def scan(
    config: str,
    output: Optional[str],
    summary_only: bool,
    ranking_method: Optional[str],
    hybrid_alpha: float,
    query_texts: Optional[str],
    rerank: bool,
    rerank_method: str,
    rerank_top_n: int,
):
    """Run a scan for adversarial hubs."""
    logger = get_logger()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Load configuration
            task = progress.add_task("Loading configuration...", total=None)
            logger.info(f"Loading configuration from {config}")
            cfg = Config.from_yaml(config)
            
            if output:
                cfg.output.out_dir = output
            
            if summary_only:
                cfg.output.out_dir = "/tmp"  # Temporary, won't be used
            
            # Override ranking config from CLI if provided
            if ranking_method:
                cfg.scan.ranking.method = ranking_method
            if hybrid_alpha is not None:
                cfg.scan.ranking.hybrid_alpha = hybrid_alpha
            if query_texts:
                cfg.scan.query_texts_path = query_texts
            
            # Override reranking config from CLI if provided
            if rerank:
                cfg.scan.ranking.rerank = True
                cfg.scan.ranking.rerank_method = rerank_method
                cfg.scan.ranking.rerank_top_n = rerank_top_n
            
            progress.update(task, completed=True)
            
            # Create scanner
            task = progress.add_task("Initializing scanner...", total=None)
            scanner = Scanner(cfg)
            progress.update(task, completed=True)
            
            # Load data
            task = progress.add_task("Loading data...", total=None)
            scanner.load_data()
            progress.update(task, completed=True)
            
            # Run scan
            task = progress.add_task("Running scan...", total=None)
            results = scanner.scan()
            progress.update(task, completed=True)
        
        # Print summary
        json_report = results["json_report"]
        summary = json_report["summary"]
        scan_info = json_report["scan_info"]
        
        # Create summary table
        table = Table(title="Scan Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Documents", f"{scan_info['num_documents']:,}")
        table.add_row("Queries Processed", f"{scan_info['num_queries']:,}")
        table.add_row("k (nearest neighbors)", str(scan_info['k']))
        table.add_row("Runtime", f"{scan_info['runtime_seconds']:.2f} seconds")
        table.add_row("Index Type", scan_info['index_type'])
        table.add_row("Metric", scan_info['metric'])
        
        console.print("\n")
        console.print(table)
        
        # Verdict summary
        verdict_table = Table(title="Verdict Summary", show_header=True, header_style="bold yellow")
        verdict_table.add_column("Verdict", style="cyan")
        verdict_table.add_column("Count", style="green")
        verdict_table.add_column("Percentage", style="blue")
        
        total = scan_info['num_documents']
        for verdict, count in summary["verdict_counts"].items():
            pct = (count / total * 100) if total > 0 else 0
            color = "red" if verdict == "HIGH" else "yellow" if verdict == "MEDIUM" else "green"
            verdict_table.add_row(
                f"[{color}]{verdict}[/{color}]",
                f"{count:,}",
                f"{pct:.2f}%"
            )
        
        console.print("\n")
        console.print(verdict_table)
        
        # Top suspicious documents
        suspicious = json_report["suspicious_documents"][:10]
        if suspicious:
            suspicious_table = Table(title="Top 10 Suspicious Documents", show_header=True, header_style="bold red")
            suspicious_table.add_column("Rank", style="cyan")
            suspicious_table.add_column("Doc Index", style="green")
            suspicious_table.add_column("Risk Score", style="yellow")
            suspicious_table.add_column("Verdict", style="red")
            suspicious_table.add_column("Hub Z-Score", style="blue")
            
            for i, doc in enumerate(suspicious, 1):
                hub_z = doc.get("hubness", {}).get("hub_z", "N/A")
                if isinstance(hub_z, (int, float)):
                    hub_z = f"{hub_z:.2f}"
                
                verdict_color = "red" if doc["verdict"] == "HIGH" else "yellow" if doc["verdict"] == "MEDIUM" else "green"
                suspicious_table.add_row(
                    str(i),
                    str(doc["doc_index"]),
                    f"{doc['risk_score']:.4f}",
                    f"[{verdict_color}]{doc['verdict']}[/{verdict_color}]",
                    str(hub_z)
                )
            
            console.print("\n")
            console.print(suspicious_table)
        
        if not summary_only:
            console.print(f"\n[bold green]Success:[/bold green] Reports saved to: [cyan]{cfg.output.out_dir}[/cyan]")
            console.print(f"  - JSON: {cfg.output.out_dir}/report.json")
            console.print(f"  - HTML: {cfg.output.out_dir}/report.html")
        
    except Exception as e:
        logger.error(f"Error during scan: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
def build_index(config: str):
    """Build a FAISS index from embeddings."""
    logger = get_logger()
    
    try:
        cfg = Config.from_yaml(config)
        
        if cfg.input.mode != "embeddings_only":
            raise ValueError("build-index command requires embeddings_only mode")
        
        scanner = Scanner(cfg)
        scanner.load_data()
        
        if cfg.index.save_path:
            click.echo(f"Index built and saved to {cfg.index.save_path}")
        else:
            click.echo("Index built but not saved (set index.save_path in config)")
        
    except Exception as e:
        logger.error(f"Error building index: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--query-texts", type=click.Path(exists=True), help="Path to query texts file")
@click.option("--methods", multiple=True, default=["vector", "hybrid"], help="Ranking methods to compare")
@click.option("--output", "-o", type=str, default="reports/comparison", help="Output directory")
def compare_ranking(config: str, query_texts: Optional[str], methods: tuple, output: str):
    """Compare detection performance across ranking methods."""
    logger = get_logger()
    
    try:
        from .sdk import compare_ranking_methods
        
        cfg = Config.from_yaml(config)
        
        if not cfg.input.embeddings_path:
            raise ValueError("embeddings_path required in config for comparison")
        
        methods_list = list(methods) if methods else ["vector", "hybrid"]
        
        console.print(f"[bold cyan]Comparing ranking methods: {', '.join(methods_list)}[/bold cyan]")
        
        comparison = compare_ranking_methods(
            embeddings_path=cfg.input.embeddings_path,
            metadata_path=cfg.input.metadata_path,
            query_texts_path=query_texts or cfg.scan.query_texts_path,
            methods=methods_list,
            output_dir=output,
            k=cfg.scan.k,
            num_queries=cfg.scan.num_queries,
        )
        
        # Display comparison table
        comparison_table = Table(title="Ranking Method Comparison", show_header=True, header_style="bold magenta")
        comparison_table.add_column("Method", style="cyan")
        comparison_table.add_column("HIGH Risk", style="red")
        comparison_table.add_column("MEDIUM Risk", style="yellow")
        comparison_table.add_column("Runtime", style="green")
        
        for method, method_results in comparison["results"].items():
            json_report = method_results["json_report"]
            verdict_counts = json_report["summary"]["verdict_counts"]
            runtime = method_results["runtime"]
            
            comparison_table.add_row(
                method,
                str(verdict_counts.get("HIGH", 0)),
                str(verdict_counts.get("MEDIUM", 0)),
                f"{runtime:.2f}s"
            )
        
        console.print("\n")
        console.print(comparison_table)
        
        if comparison.get("comparison"):
            console.print("\n[bold]Detection Metrics Comparison:[/bold]")
            metrics_table = Table(show_header=True, header_style="bold yellow")
            metrics_table.add_column("Method", style="cyan")
            metrics_table.add_column("Precision", style="green")
            metrics_table.add_column("Recall", style="green")
            metrics_table.add_column("F1", style="green")
            
            for method, metrics in comparison["comparison"].items():
                metrics_table.add_row(
                    method,
                    f"{metrics.get('precision', 0):.3f}",
                    f"{metrics.get('recall', 0):.3f}",
                    f"{metrics.get('f1', 0):.3f}",
                )
            
            console.print(metrics_table)
        
        console.print(f"\n[bold green]Comparison complete![/bold green] Results saved to: [cyan]{output}[/cyan]")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
def validate(config: str):
    """Validate approximate index against exact search."""
    logger = get_logger()
    click.echo("Validation feature coming soon...")
    # TODO: Implement validation


@cli.command()
@click.option("--doc-id", required=True, type=int, help="Document index to explain")
@click.option("--report", required=True, type=click.Path(exists=True), help="Path to JSON report")
def explain(doc_id: int, report: str):
    """Explain why a document was flagged."""
    try:
        with open(report, "r") as f:
            report_data = json.load(f)
        
        # Find document in suspicious documents
        doc_info = None
        for doc in report_data.get("suspicious_documents", []):
            if doc["doc_index"] == doc_id:
                doc_info = doc
                break
        
        if not doc_info:
            console.print(f"[bold red]Document {doc_id} not found in report[/bold red]")
            return
        
        # Create explanation table
        table = Table(title=f"Document {doc_id} Analysis", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green")
        
        verdict_color = "red" if doc_info['verdict'] == "HIGH" else "yellow" if doc_info['verdict'] == "MEDIUM" else "green"
        table.add_row("Risk Score", f"{doc_info['risk_score']:.4f}")
        table.add_row("Verdict", f"[{verdict_color}]{doc_info['verdict']}[/{verdict_color}]")
        
        if "hubness" in doc_info:
            hub = doc_info["hubness"]
            table.add_row("", "")  # Separator
            table.add_row("[bold]Hubness Metrics[/bold]", "")
            table.add_row("  Z-Score", f"{hub.get('hub_z', 'N/A'):.2f}" if isinstance(hub.get('hub_z'), (int, float)) else str(hub.get('hub_z', 'N/A')))
            hub_rate = hub.get('hub_rate')
            if hub_rate is not None:
                table.add_row("  Hub Rate", f"{hub_rate:.4f}")
            hits = hub.get('hits')
            if hits is not None:
                table.add_row("  Hits", str(hits))
        
        if "cluster_spread" in doc_info:
            table.add_row("", "")  # Separator
            table.add_row("[bold]Cluster Spread[/bold]", "")
            table.add_row("  Score", f"{doc_info['cluster_spread']['score']:.4f}")
        
        if "stability" in doc_info:
            table.add_row("", "")  # Separator
            table.add_row("[bold]Stability[/bold]", "")
            table.add_row("  Score", f"{doc_info['stability']['score']:.4f}")
        
        if "deduplication" in doc_info:
            table.add_row("", "")  # Separator
            table.add_row("[bold]Deduplication[/bold]", "")
            table.add_row("  Boilerplate Score", f"{doc_info['deduplication']['boilerplate_score']:.4f}")
        
        # Add ranking method info if available
        if "hubness" in doc_info and "ranking_method" in doc_info.get("hubness", {}):
            table.add_row("", "")  # Separator
            table.add_row("[bold]Ranking Method[/bold]", "")
            table.add_row("  Method", doc_info["hubness"].get("ranking_method", "vector"))
        
        console.print("\n")
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()

