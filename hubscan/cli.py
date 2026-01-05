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
@click.option("--hybrid-backend", type=click.Choice(["client_fusion", "native_sparse", "auto"]), default="auto", help="Hybrid search backend")
@click.option("--lexical-backend", type=click.Choice(["bm25", "tfidf"]), default="bm25", help="Lexical scoring algorithm for hybrid search")
@click.option("--text-field", type=str, default="text", help="Metadata field containing document text (for hybrid search)")
@click.option("--query-texts", type=click.Path(exists=True), help="Path to query texts file (for lexical/hybrid search)")
@click.option("--rerank", is_flag=True, help="Enable reranking as post-processing step")
@click.option("--rerank-method", type=str, default="default", help="Reranking method name (default: default)")
@click.option("--rerank-top-n", type=int, default=100, help="Number of candidates to retrieve before reranking")
@click.option("--concept-aware", is_flag=True, help="Enable concept-aware hub detection")
@click.option("--modality-aware", is_flag=True, help="Enable modality-aware hub detection")
@click.option("--concept-field", type=str, default="concept", help="Metadata field for concepts")
@click.option("--modality-field", type=str, default="modality", help="Metadata field for modality")
@click.option("--num-concepts", type=int, default=10, help="Number of concept clusters for auto-detection")
@click.option("--multi-index", is_flag=True, help="Enable multi-index mode (parallel retrieval)")
@click.option("--text-index", type=click.Path(exists=True), help="Path to text index file")
@click.option("--image-index", type=click.Path(exists=True), help="Path to image index file")
@click.option("--text-embeddings", type=click.Path(exists=True), help="Path to text embeddings file")
@click.option("--image-embeddings", type=click.Path(exists=True), help="Path to image embeddings file")
@click.option("--late-fusion", is_flag=True, help="Enable late fusion of multi-index results")
@click.option("--fusion-method", type=click.Choice(["rrf", "weighted_sum", "max"]), default="rrf", help="Late fusion method")
@click.option("--text-weight", type=float, default=0.4, help="Weight for text index in fusion")
@click.option("--image-weight", type=float, default=0.4, help="Weight for image index in fusion")
def scan(
    config: str,
    output: Optional[str],
    summary_only: bool,
    ranking_method: Optional[str],
    hybrid_alpha: float,
    hybrid_backend: str,
    lexical_backend: str,
    text_field: str,
    query_texts: Optional[str],
    rerank: bool,
    rerank_method: str,
    rerank_top_n: int,
    concept_aware: bool,
    modality_aware: bool,
    concept_field: str,
    modality_field: str,
    num_concepts: int,
    multi_index: bool,
    text_index: Optional[str],
    image_index: Optional[str],
    text_embeddings: Optional[str],
    image_embeddings: Optional[str],
    late_fusion: bool,
    fusion_method: str,
    text_weight: float,
    image_weight: float,
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
            
            # Override hybrid search config from CLI
            if hybrid_backend:
                cfg.scan.ranking.hybrid.backend = hybrid_backend
            if lexical_backend:
                cfg.scan.ranking.hybrid.lexical_backend = lexical_backend
            if text_field:
                cfg.scan.ranking.hybrid.text_field = text_field
            
            # Override reranking config from CLI if provided
            if rerank:
                cfg.scan.ranking.rerank = True
                cfg.scan.ranking.rerank_method = rerank_method
                cfg.scan.ranking.rerank_top_n = rerank_top_n
            
            # Override concept-aware and modality-aware detection from CLI
            if concept_aware:
                cfg.detectors.concept_aware.enabled = True
                cfg.detectors.concept_aware.metadata_field = concept_field
                cfg.detectors.concept_aware.num_concepts = num_concepts
            if modality_aware:
                cfg.detectors.modality_aware.enabled = True
                cfg.detectors.modality_aware.doc_modality_field = modality_field
                cfg.detectors.modality_aware.query_modality_field = modality_field
            
            # Override multi-index configuration from CLI
            if multi_index or text_index or image_index:
                cfg.input.mode = "multi_index"
                if cfg.input.multi_index is None:
                    from .config import MultiIndexConfig
                    cfg.input.multi_index = MultiIndexConfig()
                if text_index:
                    cfg.input.multi_index.text_index_path = text_index
                if image_index:
                    cfg.input.multi_index.image_index_path = image_index
                if text_embeddings:
                    cfg.input.multi_index.text_embeddings_path = text_embeddings
                if image_embeddings:
                    cfg.input.multi_index.image_embeddings_path = image_embeddings
                
                # Enable parallel retrieval
                cfg.scan.ranking.parallel_retrieval = True
            
            # Override late fusion configuration from CLI
            if late_fusion:
                if cfg.input.late_fusion is None:
                    from .config import LateFusionConfig
                    cfg.input.late_fusion = LateFusionConfig()
                cfg.input.late_fusion.enabled = True
                cfg.input.late_fusion.fusion_method = fusion_method
                cfg.input.late_fusion.text_weight = text_weight
                cfg.input.late_fusion.image_weight = image_weight
            
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
            
            # Add concept/modality columns if enabled
            show_concept = concept_aware or cfg.detectors.concept_aware.enabled
            show_modality = modality_aware or cfg.detectors.modality_aware.enabled
            if show_concept:
                suspicious_table.add_column("Concept Z", style="magenta")
            if show_modality:
                suspicious_table.add_column("Cross-Modal", style="cyan")
            
            for i, doc in enumerate(suspicious, 1):
                hub_z = doc.get("hubness", {}).get("hub_z", "N/A")
                if isinstance(hub_z, (int, float)):
                    hub_z = f"{hub_z:.2f}"
                
                verdict_color = "red" if doc["verdict"] == "HIGH" else "yellow" if doc["verdict"] == "MEDIUM" else "green"
                
                row = [
                    str(i),
                    str(doc["doc_index"]),
                    f"{doc['risk_score']:.4f}",
                    f"[{verdict_color}]{doc['verdict']}[/{verdict_color}]",
                    str(hub_z)
                ]
                
                if show_concept:
                    concept_z = doc.get("hubness", {}).get("max_concept_hub_z", "N/A")
                    if isinstance(concept_z, (int, float)):
                        concept_z = f"{concept_z:.2f}"
                    row.append(str(concept_z))
                
                if show_modality:
                    cross_modal = doc.get("hubness", {}).get("cross_modal_ratio", "N/A")
                    if isinstance(cross_modal, (int, float)):
                        cross_modal = f"{cross_modal:.2f}"
                    row.append(str(cross_modal))
                
                suspicious_table.add_row(*row)
            
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
@click.option("--output", "-o", required=True, type=str, help="Output path for embeddings (.npy file)")
@click.option("--batch-size", default=1000, type=int, help="Number of vectors to retrieve per batch")
@click.option("--limit", default=None, type=int, help="Maximum number of vectors to extract (None = all)")
def extract_embeddings(config: str, output: str, batch_size: int, limit: Optional[int]):
    """Extract embeddings from a vector database."""
    logger = get_logger()
    
    try:
        config_obj = Config.from_yaml(config)
        scanner = Scanner(config_obj)
        scanner.load_data()
        
        embeddings, ids = scanner.extract_embeddings(
            output_path=output,
            batch_size=batch_size,
            limit=limit,
        )
        
        console.print(f"[green]Successfully extracted {len(embeddings)} embeddings[/green]")
        console.print(f"[green]Saved to: {output}[/green]")
        if ids:
            console.print(f"[green]Extracted {len(ids)} document IDs[/green]")
        
    except Exception as e:
        logger.error(f"Failed to extract embeddings: {e}", exc_info=True)
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--check-data", is_flag=True, help="Verify that data files exist and are readable")
@click.option("--check-index", is_flag=True, help="Validate index integrity and perform test queries")
@click.option("--sample-queries", type=int, default=10, help="Number of sample queries for index validation")
def validate(config: str, check_data: bool, check_index: bool, sample_queries: int):
    """Validate a configuration file and optionally verify data/index integrity."""
    logger = get_logger()
    
    all_valid = True
    
    try:
        # Load and validate config structure
        console.print(f"[bold cyan]Validating configuration: {config}[/bold cyan]\n")
        cfg = Config.from_yaml(config)
        console.print(f"[green]Config structure is valid[/green]")
        
        # Display configuration summary
        table = Table(title="Configuration Summary", show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Input Mode", cfg.input.mode)
        table.add_row("Metric", cfg.input.metric)
        table.add_row("k (neighbors)", str(cfg.scan.k))
        table.add_row("num_queries", str(cfg.scan.num_queries))
        table.add_row("Ranking Method", cfg.scan.ranking.method)
        table.add_row("Output Directory", cfg.output.out_dir)
        
        # Show enabled detectors
        enabled_detectors = []
        if cfg.detectors.hubness.enabled:
            enabled_detectors.append("hubness")
        if cfg.detectors.cluster_spread.enabled:
            enabled_detectors.append("cluster_spread")
        if cfg.detectors.stability.enabled:
            enabled_detectors.append("stability")
        if cfg.detectors.dedup.enabled:
            enabled_detectors.append("dedup")
        table.add_row("Enabled Detectors", ", ".join(enabled_detectors) or "none")
        
        console.print("\n")
        console.print(table)
        
        if check_data:
            console.print("\n[bold cyan]Checking data files...[/bold cyan]")
            
            # Check embeddings path
            if cfg.input.embeddings_path:
                path = Path(cfg.input.embeddings_path)
                if path.exists():
                    try:
                        import numpy as np
                        data = np.load(path)
                        console.print(f"  [green]Embeddings: {path}[/green]")
                        console.print(f"    Shape: {data.shape}, dtype: {data.dtype}")
                    except Exception as e:
                        console.print(f"  [red]Embeddings: {path} (LOAD ERROR: {e})[/red]")
                        all_valid = False
                else:
                    console.print(f"  [red]Embeddings: {path} (NOT FOUND)[/red]")
                    all_valid = False
            
            # Check index path
            if cfg.input.index_path:
                path = Path(cfg.input.index_path)
                if path.exists():
                    try:
                        import faiss
                        index = faiss.read_index(str(path))
                        console.print(f"  [green]Index: {path}[/green]")
                        console.print(f"    Vectors: {index.ntotal}, dimension: {index.d}")
                    except Exception as e:
                        console.print(f"  [red]Index: {path} (LOAD ERROR: {e})[/red]")
                        all_valid = False
                else:
                    console.print(f"  [red]Index: {path} (NOT FOUND)[/red]")
                    all_valid = False
            
            # Check metadata path
            if cfg.input.metadata_path:
                path = Path(cfg.input.metadata_path)
                if path.exists():
                    try:
                        import json
                        with open(path) as f:
                            metadata = json.load(f)
                        if isinstance(metadata, list):
                            console.print(f"  [green]Metadata: {path}[/green]")
                            console.print(f"    Records: {len(metadata)}")
                            if metadata:
                                console.print(f"    Fields: {list(metadata[0].keys())[:5]}...")
                        elif isinstance(metadata, dict):
                            console.print(f"  [green]Metadata: {path}[/green]")
                            console.print(f"    Fields: {list(metadata.keys())[:5]}...")
                    except Exception as e:
                        console.print(f"  [red]Metadata: {path} (LOAD ERROR: {e})[/red]")
                        all_valid = False
                else:
                    console.print(f"  [red]Metadata: {path} (NOT FOUND)[/red]")
                    all_valid = False
            
            # Check query texts path
            if cfg.scan.query_texts_path:
                path = Path(cfg.scan.query_texts_path)
                if path.exists():
                    try:
                        import json
                        with open(path) as f:
                            texts = json.load(f)
                        console.print(f"  [green]Query Texts: {path}[/green]")
                        console.print(f"    Count: {len(texts)}")
                    except Exception as e:
                        console.print(f"  [red]Query Texts: {path} (LOAD ERROR: {e})[/red]")
                        all_valid = False
                else:
                    console.print(f"  [yellow]Query Texts: {path} (NOT FOUND - optional)[/yellow]")
        
        if check_index:
            console.print("\n[bold cyan]Validating index integrity...[/bold cyan]")
            
            try:
                # Load scanner to test index
                scanner = Scanner(cfg)
                scanner.load_data()
                
                if scanner.index is not None and scanner.doc_embeddings is not None:
                    import numpy as np
                    
                    # Run sample queries
                    num_docs = len(scanner.doc_embeddings)
                    num_test = min(sample_queries, num_docs)
                    test_indices = np.random.choice(num_docs, num_test, replace=False)
                    test_queries = scanner.doc_embeddings[test_indices]
                    
                    console.print(f"  Running {num_test} test queries...")
                    
                    # Search
                    distances, indices = scanner.index.search(test_queries, cfg.scan.k)
                    
                    # Validate results
                    valid_results = 0
                    self_matches = 0
                    for i, test_idx in enumerate(test_indices):
                        if test_idx in indices[i]:
                            self_matches += 1
                        if np.all(indices[i] >= 0) and np.all(indices[i] < num_docs):
                            valid_results += 1
                    
                    console.print(f"  [green]Valid result sets: {valid_results}/{num_test}[/green]")
                    console.print(f"  [green]Self-match rate: {self_matches}/{num_test}[/green]")
                    
                    if self_matches < num_test * 0.9:
                        console.print(f"  [yellow]Warning: Low self-match rate may indicate index issues[/yellow]")
                else:
                    console.print(f"  [yellow]Skipping index validation (no index or embeddings loaded)[/yellow]")
                    
            except Exception as e:
                console.print(f"  [red]Index validation failed: {e}[/red]")
                all_valid = False
        
        # Final verdict
        console.print("\n")
        if all_valid:
            console.print(f"[bold green]Validation PASSED[/bold green]")
        else:
            console.print(f"[bold red]Validation FAILED - see errors above[/bold red]")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}", exc_info=True)
        console.print(f"[bold red]Config validation failed: {e}[/bold red]")
        sys.exit(1)


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

