"""
pngToAgent.py: A comprehensive pipeline for extracting spectroscopic intelligence
from high-resolution PNG images of scientific documents via OpenAI's multimodal
structured outputs, with AI-powered factorization into Obsidian knowledge vaults.

Architecture:
- Environment & utilities for image encoding
- Pydantic schemas for structured extraction
- OpenAI multimodal vision analysis with gpt-4o
- Numerical peak fitting stub (scipy/lmfit integration point)
- Obsidian markdown generation with auto-generated wiki-links
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI, APIError
from pydantic import BaseModel, Field

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION & ENVIRONMENT
# ============================================================================

def load_environment() -> str:
    """Load OpenAI API key from info.env using python-dotenv."""
    load_dotenv("info.env")
    api_key = os.getenv("SUPER_SECRET_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "SUPER_SECRET_API_KEY not found in info.env. "
            "Please ensure info.env contains the OpenAI API key."
        )
    return api_key


# ============================================================================
# PYDANTIC SCHEMAS FOR STRUCTURED EXTRACTION
# ============================================================================

class DocumentMetadata(BaseModel):
    """Metadata extracted from the document."""

    title: str = Field(
        ...,
        description="Title of the paper or document.",
    )
    author: str = Field(
        ...,
        description="Primary author(s) of the document.",
    )
    year: int = Field(
        ...,
        description="Publication year.",
    )
    page_source: str = Field(
        ...,
        description="Source page identifier (e.g., 'page_3_of_10').",
    )
    institution: Optional[str] = Field(
        None,
        description="Affiliated institution if visible.",
    )


class DetectedPeak(BaseModel):
    """A single detected spectroscopic peak."""

    chemical_assignment: str = Field(
        ...,
        description="Chemical or functional group assignment (e.g., 'C=O', 'N-H').",
    )
    estimated_position_ev: float = Field(
        ...,
        description="Estimated peak position in eV (or wavenumber for IR/Raman).",
    )
    fitting_bounds: List[float] = Field(
        ...,
        description="Lower and upper bounds for numerical curve fitting. Provide exactly two values: [lower, upper].",
    )
    intensity_estimate: Optional[str] = Field(
        None,
        description="Qualitative intensity estimate: 'strong', 'medium', 'weak'.",
    )
    notes: Optional[str] = Field(
        None,
        description="Additional observations (e.g., 'doublet', 'shoulder', 'satellite').",
    )


class ContextualSummary(BaseModel):
    """Extracted textual and methodological context from the document."""

    sample_description: str = Field(
        ...,
        description="Description of the sample(s) analyzed.",
    )
    experimental_technique: str = Field(
        ...,
        description="Spectroscopic technique used (e.g., 'XPS', 'FTIR', 'Raman').",
    )
    chemical_species: List[str] = Field(
        default_factory=list,
        description="List of identified chemical species or compounds.",
    )
    methodology_notes: Optional[str] = Field(
        None,
        description="Key experimental parameters, instrument settings, or conditions.",
    )
    findings_summary: str = Field(
        ...,
        description="Brief summary of key findings and interpretations from the page.",
    )


class SpectroscopyAnalysis(BaseModel):
    """Complete structured extraction from a spectroscopic document page."""

    document_metadata: DocumentMetadata
    contextual_summary: ContextualSummary
    detected_peaks: List[DetectedPeak] = Field(
        default_factory=list,
        description="List of detected spectroscopic peaks.",
    )
    extraction_confidence: float = Field(
        ...,
        description="Confidence score (0.0 to 1.0) for this extraction.",
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Any caveats or limitations in the analysis.",
    )


# ============================================================================
# IMAGE ENCODING & FILE HANDLING
# ============================================================================

def encode_image_to_base64(image_path: Path) -> str:
    """Convert a PNG image file to a base64-encoded string."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")

    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_sorted_png_files(processed_dir: Path) -> List[Path]:
    """
    Retrieve PNG files from processed_dir, sorted by page number.
    Attempts to extract page numbers from filenames (e.g., page-001, p3, etc.).
    """
    if not processed_dir.exists() or not processed_dir.is_dir():
        logger.warning(f"Processed directory does not exist: {processed_dir}")
        return []

    png_files = sorted(processed_dir.glob("*.png"))

    def extract_page_number(filename: str) -> Tuple[int, str]:
        """Extract numeric page indicator from filename."""
        # Try patterns like "page-001", "p03", "3-of-20", etc.
        match = re.search(r"(\d+)", filename)
        if match:
            return (int(match.group(1)), filename)
        return (float("inf"), filename)

    png_files.sort(key=lambda p: extract_page_number(p.name))
    return png_files


# ============================================================================
# OPENAI MULTIMODAL STRUCTURED EXTRACTION
# ============================================================================

def extract_spectroscopy_data(
    client: OpenAI,
    image_path: Path,
    model: str = "gpt-4o",
) -> SpectroscopyAnalysis:
    """
    Call OpenAI's gpt-4o with vision capability to extract spectroscopic
    intelligence from a PNG image using structured JSON output.

    Args:
        client: Initialized OpenAI client
        image_path: Path to the PNG image
        model: Model identifier (default: gpt-4o)

    Returns:
        SpectroscopyAnalysis: Parsed structured extraction
    """
    logger.info(f"Extracting spectroscopy data from {image_path.name}...")

    image_base64 = encode_image_to_base64(image_path)

    system_prompt = """You are an expert spectroscopist and data scientist specializing
in the analysis of legacy scientific publications. Your task is to:

1. Extract document metadata (title, author, year, page info).
2. Identify the experimental technique and sample description.
3. Detect all visible spectroscopic peaks, their positions, and chemical assignments.
4. Provide a concise, factually grounded summary of findings.
5. Use precise chemical nomenclature and physical units (eV, cm⁻¹, ppm, etc.).

Be conservative in your confidence scores. If data is ambiguous or partially visible,
note this explicitly and lower the confidence score accordingly."""

    user_prompt = """Please analyze this spectroscopic document page and extract:
- Document metadata (title, author, year, publication context)
- Experimental technique and sample details
- All detected spectroscopic peaks with chemical assignments
- Estimated peak positions (in appropriate units)
- A brief summary of key findings

Ensure peak positions are reasonable for the stated technique and provide fitting bounds
for potential numerical curve fitting."""

    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                            },
                        },
                    ],
                },
            ],
            response_format=SpectroscopyAnalysis,
            temperature=0.3,
            max_tokens=2000,
        )

        extracted = response.choices[0].message.parsed
        if extracted is None:
            raise ValueError("OpenAI returned null parsed response")

        logger.info(
            f"Successfully extracted data for '{extracted.document_metadata.title}' "
            f"(confidence: {extracted.extraction_confidence:.2f})"
        )
        return extracted

    except APIError as e:
        logger.error(f"OpenAI API error while processing {image_path.name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error extracting data from {image_path.name}: {e}")
        raise


# ============================================================================
# NUMERICAL PEAK FITTING STUB
# ============================================================================

class FitResult(BaseModel):
    """Result of numerical peak fitting."""

    chemical_assignment: str
    fitted_position: float
    fitted_amplitude: float
    fitted_width: float
    r_squared: float
    notes: str


def perform_numerical_fit(
    peak_data: List[DetectedPeak],
) -> List[FitResult]:
    """
    Placeholder for numerical peak fitting using scipy.optimize or lmfit.

    In production, this would:
    - Load or retrieve actual spectroscopic data (CSV, HDF5, etc.)
    - Fit Gaussian, Lorentzian, or other lineshapes to raw data
    - Use the bounds from peak_data to constrain fitting
    - Return goodness-of-fit statistics (R², chi-squared)

    For now, returns a stub with estimated fit parameters.
    """
    fits = []
    for peak in peak_data:
        # Stub: use estimated position as the fit center
        fit = FitResult(
            chemical_assignment=peak.chemical_assignment,
            fitted_position=peak.estimated_position_ev,
            fitted_amplitude=0.75,  # Placeholder
            fitted_width=(peak.fitting_bounds[1] - peak.fitting_bounds[0]) / 2.355,
            r_squared=0.95,  # Placeholder
            notes="Stub fit pending integration with raw spectroscopic data.",
        )
        fits.append(fit)

    return fits


# ============================================================================
# OBSIDIAN VAULT GENERATION & AI REFACTORING
# ============================================================================

class ObsidianRefactoringAgent:
    """
    AI-powered agent that transforms raw spectroscopic JSON into a beautifully
    factorized Obsidian markdown note optimized for knowledge vault integration.
    """

    def __init__(self, vault_dir: Path):
        """Initialize the refactoring agent with a target vault directory."""
        self.vault_dir = Path(vault_dir)
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    def generate_wiki_links(self, analysis: SpectroscopyAnalysis) -> List[str]:
        """
        Auto-generate Obsidian wiki-links based on extracted content.
        Identifies chemical species, techniques, and concepts for internal linking.
        """
        links = set()

        # Add chemical species as links
        for species in analysis.contextual_summary.chemical_species:
            # Convert to title case and sanitize
            link_text = species.strip().replace(" ", " ")
            links.add(f"[[{link_text}]]")

        # Add spectroscopic technique
        technique = analysis.contextual_summary.experimental_technique
        links.add(f"[[{technique} Spectroscopy]]")

        # Add chemical functional groups from peaks
        for peak in analysis.detected_peaks:
            assignment = peak.chemical_assignment.strip()
            # Skip single characters or generic assignments
            if len(assignment) > 2 and assignment.lower() not in {"c", "n", "o", "h", "s"}:
                links.add(f"[[{assignment}]]")

        # Add methodology concepts
        if analysis.contextual_summary.methodology_notes:
            methodology = analysis.contextual_summary.methodology_notes
            if "X-ray" in methodology or "XPS" in methodology:
                links.add("[[X-ray Photoelectron Spectroscopy]]")
            if "IR" in methodology or "Infrared" in methodology:
                links.add("[[Infrared Spectroscopy]]")
            if "Raman" in methodology:
                links.add("[[Raman Spectroscopy]]")
            if "UV" in methodology or "Vis" in methodology:
                links.add("[[UV-Vis Spectroscopy]]")

        return sorted(list(links))

    def generate_yaml_frontmatter(
        self,
        analysis: SpectroscopyAnalysis,
        wiki_links: List[str],
    ) -> str:
        """Generate YAML frontmatter for the Obsidian note."""
        frontmatter = "---\n"

        # Title
        frontmatter += f"title: {analysis.document_metadata.title}\n"

        # Tags (from technique and chemical species)
        tags = ["spectroscopy"]
        tags.append(analysis.contextual_summary.experimental_technique.lower().replace(" ", "-"))
        for species in analysis.contextual_summary.chemical_species[:3]:  # Limit to 3
            sanitized = species.lower().replace(" ", "-").replace("/", "")
            tags.append(sanitized)
        frontmatter += f"tags: [{', '.join(tags)}]\n"

        # Source metadata
        frontmatter += f"source: {analysis.document_metadata.author}, {analysis.document_metadata.year}\n"
        frontmatter += f"page: {analysis.document_metadata.page_source}\n"

        # Elements/Chemical species
        elements = ", ".join(analysis.contextual_summary.chemical_species[:5])
        frontmatter += f"elements: {elements}\n"

        # Date extracted
        frontmatter += f"extracted: {datetime.now().isoformat()}\n"

        # Confidence
        frontmatter += f"extraction_confidence: {analysis.extraction_confidence:.2f}\n"

        # Related topics (wiki-links as a structured list)
        if wiki_links:
            frontmatter += f"related:\n"
            for link in wiki_links[:10]:  # Limit to 10 links
                # Extract link text without brackets
                link_text = link.strip("[]")
                frontmatter += f"  - {link_text}\n"

        frontmatter += "---\n\n"
        return frontmatter

    def generate_peaks_table(self, peaks: List[DetectedPeak]) -> str:
        """Generate a markdown table of detected peaks."""
        if not peaks:
            return ""

        table = "## Detected Peaks\n\n"
        table += "| Chemical Assignment | Position (eV/cm⁻¹) | Lower Bound | Upper Bound | Intensity | Notes |\n"
        table += "|---|---|---|---|---|---|\n"

        for peak in peaks:
            lower, upper = peak.fitting_bounds
            intensity = peak.intensity_estimate or "N/A"
            notes = peak.notes or ""
            table += (
                f"| {peak.chemical_assignment} | {peak.estimated_position_ev:.2f} "
                f"| {lower:.2f} | {upper:.2f} | {intensity} | {notes} |\n"
            )

        table += "\n"
        return table

    def generate_markdown_content(self, analysis: SpectroscopyAnalysis) -> str:
        """Generate the core markdown content with all sections."""
        content = ""

        # Title heading
        content += f"# {analysis.document_metadata.title}\n\n"

        # Source attribution
        content += (
            f"**Source:** {analysis.document_metadata.author}, "
            f"{analysis.document_metadata.year}  \n"
            f"**Page:** {analysis.document_metadata.page_source}  \n"
            f"**Technique:** {analysis.contextual_summary.experimental_technique}  \n\n"
        )

        # Sample description section
        content += "## Sample & Methodology\n\n"
        content += f"{analysis.contextual_summary.sample_description}\n\n"
        if analysis.contextual_summary.methodology_notes:
            content += f"**Experimental Parameters:**\n\n{analysis.contextual_summary.methodology_notes}\n\n"

        # Key findings section
        content += "## Key Findings\n\n"
        content += f"{analysis.contextual_summary.findings_summary}\n\n"

        # Peaks table
        content += self.generate_peaks_table(analysis.detected_peaks)

        # Chemical species section
        if analysis.contextual_summary.chemical_species:
            content += "## Chemical Species Identified\n\n"
            for species in analysis.contextual_summary.chemical_species:
                content += f"- [[{species}]]\n"
            content += "\n"

        # Extraction metadata
        content += "## Extraction Metadata\n\n"
        content += f"- **Confidence Score:** {analysis.extraction_confidence:.2%}\n"
        if analysis.extraction_notes:
            content += f"- **Notes:** {analysis.extraction_notes}\n"
        content += f"- **Extracted:** {datetime.now().isoformat()}\n\n"

        return content

    def create_obsidian_note(self, analysis: SpectroscopyAnalysis) -> Path:
        """
        Generate a complete Obsidian markdown file from spectroscopy analysis.
        Returns the path to the created file.
        """
        # Generate wiki links
        wiki_links = self.generate_wiki_links(analysis)

        # Generate YAML frontmatter
        frontmatter = self.generate_yaml_frontmatter(analysis, wiki_links)

        # Generate markdown content
        content = self.generate_markdown_content(analysis)

        # Combine sections
        full_content = frontmatter + content

        # Add wiki-links section
        if wiki_links:
            full_content += "## Related Topics\n\n"
            for link in wiki_links:
                full_content += f"{link}\n"

        # Create filename from document metadata
        safe_title = (
            analysis.document_metadata.title.lower()
            .replace(" ", "-")
            .replace("/", "-")
            .replace(":", "")[:50]
        )
        filename = f"{safe_title}_{analysis.document_metadata.page_source}.md"
        output_path = self.vault_dir / filename

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        logger.info(f"Obsidian note created: {output_path}")
        return output_path


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def process_image_batch(
    processed_dir: Path = Path("Deposit/processed"),
    vault_dir: Path = Path("ObsidianVault"),
    output_dir: Path = Path("ExtractionOutput"),
) -> Dict[str, Any]:
    """
    Main orchestration function: process all PNG images and generate Obsidian notes.

    Args:
        processed_dir: Directory containing processed PNG files
        vault_dir: Output directory for Obsidian markdown files
        output_dir: Directory for JSON extraction outputs

    Returns:
        Summary dictionary of processing results
    """
    # Initialize
    api_key = load_environment()
    client = OpenAI(api_key=api_key)
    refactoring_agent = ObsidianRefactoringAgent(vault_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get PNG files
    png_files = get_sorted_png_files(processed_dir)
    if not png_files:
        logger.warning(f"No PNG files found in {processed_dir}")
        return {
            "status": "no_files",
            "processed_count": 0,
            "errors": ["No PNG files found in processed directory"],
        }

    logger.info(f"Found {len(png_files)} PNG files to process")

    results = {
        "status": "success",
        "processed_count": 0,
        "successful": [],
        "failed": [],
        "notes_created": [],
        "errors": [],
    }

    # Process each image
    for image_path in png_files:
        try:
            logger.info(f"\nProcessing: {image_path.name}")

            # Extract spectroscopy data from image
            analysis = extract_spectroscopy_data(client, image_path)

            # Save JSON output
            json_output_path = output_dir / f"{image_path.stem}_analysis.json"
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(analysis.model_dump(), f, indent=2)

            logger.info(f"JSON analysis saved: {json_output_path}")

            # Perform numerical peak fitting (stub)
            fit_results = perform_numerical_fit(analysis.detected_peaks)
            logger.info(f"Performed fitting on {len(fit_results)} peaks")

            # Generate Obsidian note
            note_path = refactoring_agent.create_obsidian_note(analysis)

            results["processed_count"] += 1
            results["successful"].append(image_path.name)
            results["notes_created"].append(str(note_path))

        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {str(e)}"
            logger.error(error_msg)
            results["failed"].append(image_path.name)
            results["errors"].append(error_msg)

    # Summary logging
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total processed: {results['processed_count']}")
    logger.info(f"Successful: {len(results['successful'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    logger.info(f"Obsidian notes created: {len(results['notes_created'])}")

    if results["notes_created"]:
        logger.info(f"\nObsidian vault directory: {vault_dir}")
        logger.info(f"JSON outputs directory: {output_dir}")

    return results


# ============================================================================
# ENTRY POINT WITH STARTUP MODE SELECTION
# ============================================================================

def display_startup_menu() -> str:
    """Display startup menu and return selected mode."""
    print("\n" + "=" * 70)
    print("SPECTROSCOPY ANALYSIS PIPELINE - STARTUP MENU")
    print("=" * 70)
    print("\nSelect analysis mode:\n")
    print("  [1] DEFAULT RUNTIME")
    print("     - Analyze entire pages as single entities")
    print("     - Fast processing, minimal user interaction\n")
    print("  [2] GRAPH SEPARATION WITH VALIDATION")
    print("     - Detect & separate multiple graphs per page")
    print("     - Manual confirmation for each detected graph")
    print("     - Edit bounding boxes interactively\n")
    print("  [3] GRAPH SEPARATION WITH VISUALIZATION")
    print("     - Detect & separate multiple graphs per page")
    print("     - Visual preview + manual validation for each graph")
    print("     - Full interactive bounding box editor\n")
    
    while True:
        choice = input("Enter your choice [1/2/3]: ").strip()
        if choice in ["1", "2", "3"]:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Process PNG images from scientific documents with graph-centric or "
            "page-centric analysis. Extract spectroscopic intelligence via OpenAI "
            "vision and generate Obsidian knowledge vault notes."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="Deposit/processed",
        help="Directory containing processed PNG files.",
    )
    parser.add_argument(
        "--vault-dir",
        type=str,
        default="ObsidianVault",
        help="Output directory for Obsidian markdown files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ExtractionOutput",
        help="Directory for JSON extraction outputs.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["default", "validate", "visualize"],
        default=None,
        help="Analysis mode: 'default' (page-centric), 'validate' (graph separation), or 'visualize' (with preview).",
    )
    parser.add_argument(
        "--skip-menu",
        action="store_true",
        help="Skip startup menu and use --mode or default.",
    )

    args = parser.parse_args()

    # Determine mode
    if args.skip_menu or args.mode:
        mode_choice = {
            "default": "1",
            "validate": "2",
            "visualize": "3",
        }.get(args.mode, "1")
    else:
        mode_choice = display_startup_menu()

    # Route to appropriate pipeline
    if mode_choice == "1":
        # Default page-centric analysis
        logger.info("Running DEFAULT RUNTIME mode (page-centric analysis)")
        summary = process_image_batch(
            processed_dir=Path(args.processed_dir),
            vault_dir=Path(args.vault_dir),
            output_dir=Path(args.output_dir),
        )
    else:
        # Graph-centric analysis with validation/visualization
        try:
            from pngSeperation import process_image_batch as process_graphs
            
            validate_mode = mode_choice == "2"
            visualize_mode = mode_choice == "3"
            
            if visualize_mode:
                logger.info("Running GRAPH SEPARATION with VISUALIZATION mode")
            else:
                logger.info("Running GRAPH SEPARATION with VALIDATION mode")
            
            summary = process_graphs(
                processed_dir=Path(args.processed_dir),
                output_dir=Path(args.output_dir),
                vault_dir=Path(args.vault_dir),
                validate_crops=True,
                enable_visualization=visualize_mode,
            )
        except ImportError:
            logger.error("pngSeperation module not found. Falling back to DEFAULT RUNTIME.")
            summary = process_image_batch(
                processed_dir=Path(args.processed_dir),
                vault_dir=Path(args.vault_dir),
                output_dir=Path(args.output_dir),
            )

    # Exit with appropriate code
    sys.exit(0 if summary["status"] == "success" else 1)
