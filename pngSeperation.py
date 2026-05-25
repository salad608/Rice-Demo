"""
pngSeperation.py: Graph-Centric Spectroscopy Analysis Pipeline

Refactored from page-centric to graph-centric architecture:
- Detects multiple spectroscopy graphs within a single PNG page
- Provides interactive validation of auto-detected crops
- Stores data in hierarchical folder structure (page → graphs)
- Performs targeted AI analysis per graph with full page context
- Generates Obsidian vault entries with graph-level granularity

Architecture:
- Graph detection using OpenCV edge/contour analysis
- Interactive terminal validation with matplotlib display
- Folder-per-page storage with per-graph JSON/markdown outputs
- Dual-phase OpenAI calls: page context + graph-specific analysis
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, APIError
from PIL import Image
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
# DATA CLASSES FOR GRAPH DETECTION
# ============================================================================

@dataclass
class DetectedGraph:
    """Represents a detected graph region within a page."""

    index: int
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    crop_image: np.ndarray

    def get_crop_size_mb(self) -> float:
        """Estimate crop size in MB for logging."""
        return self.crop_image.nbytes / (1024 * 1024)


@dataclass
class PageContext:
    """Extracted context from the entire page."""

    title: str
    author: str
    year: int
    sample_description: str
    experimental_technique: str
    chemical_species: List[str]
    methodology_notes: str
    overall_findings: str


# ============================================================================
# PYDANTIC SCHEMAS FOR STRUCTURED EXTRACTION
# ============================================================================

class DocumentMetadata(BaseModel):
    """Metadata extracted from the document page."""

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
    """A single detected spectroscopic peak in a specific graph."""

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


class GraphAnalysis(BaseModel):
    """Analysis of a single detected graph/plot."""

    graph_index: int = Field(
        ...,
        description="Sequential index of this graph within the page (1-indexed).",
    )
    graph_title: Optional[str] = Field(
        None,
        description="Title or label visible on the graph.",
    )
    detected_peaks: List[DetectedPeak] = Field(
        default_factory=list,
        description="List of detected spectroscopic peaks in this graph.",
    )
    axis_labels: Optional[Dict[str, str]] = Field(
        None,
        description="Detected axis labels (e.g., {'x': 'Wavenumber (cm-1)', 'y': 'Intensity'}).",
    )
    graph_description: str = Field(
        ...,
        description="Detailed description of what this graph shows.",
    )
    extraction_confidence: float = Field(
        ...,
        description="Confidence score (0.0 to 1.0) for this graph extraction.",
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Any caveats or limitations in the analysis.",
    )


class PageAnalysisSummary(BaseModel):
    """Summary of overall page context and findings."""

    document_metadata: DocumentMetadata
    sample_description: str
    experimental_technique: str
    chemical_species: List[str]
    methodology_notes: str
    overall_findings: str
    page_extraction_confidence: float


# ============================================================================
# GRAPH DETECTION & SEPARATION MODULE
# ============================================================================

class GraphDetector:
    """
    Detects and separates spectroscopy graphs from page PNG images
    using OpenCV contour and edge detection.
    """

    def __init__(
        self,
        min_area: int = 10000,
        max_area_ratio: float = 0.95,
        edge_threshold1: int = 50,
        edge_threshold2: int = 150,
    ):
        """
        Initialize the graph detector.

        Args:
            min_area: Minimum area in pixels for a detected region to be considered a graph
            max_area_ratio: Maximum ratio of region area to total image area (filters out full page)
            edge_threshold1: Lower threshold for Canny edge detection
            edge_threshold2: Upper threshold for Canny edge detection
        """
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.edge_threshold1 = edge_threshold1
        self.edge_threshold2 = edge_threshold2

    def detect_graphs(self, image_path: Path) -> List[DetectedGraph]:
        """
        Detect graph regions in a page PNG and return cropped images.

        Args:
            image_path: Path to the PNG file

        Returns:
            List of DetectedGraph objects with bounding boxes and crop images
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        total_area = height * width

        # Edge detection
        edges = cv2.Canny(gray, self.edge_threshold1, self.edge_threshold2)

        # Morphological operations to enhance connected components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_graphs: List[DetectedGraph] = []

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            area_ratio = area / total_area

            # Filter by area constraints
            if area < self.min_area or area_ratio > self.max_area_ratio:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Add padding to the bounding box
            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(width - x, w + 2 * pad)
            h = min(height - y, h + 2 * pad)

            # Crop the image
            crop = image[y : y + h, x : x + w]

            confidence = min(1.0, area / (self.min_area * 10))

            detected_graph = DetectedGraph(
                index=len(detected_graphs),
                bbox=(x, y, w, h),
                confidence=confidence,
                crop_image=crop,
            )
            detected_graphs.append(detected_graph)

        logger.info(f"Detected {len(detected_graphs)} graph(s) in {image_path.name}")
        return detected_graphs

    def visualize_crops(self, graphs: List[DetectedGraph]) -> None:
        """
        Display detected graph crops using matplotlib for interactive validation.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping visualization")
            return

        for graph in graphs:
            # Convert BGR to RGB for display
            rgb_crop = cv2.cvtColor(graph.crop_image, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(12, 6))
            plt.imshow(rgb_crop)
            plt.title(f"Detected Graph #{graph.index + 1} (Confidence: {graph.confidence:.2f})")
            plt.axis("off")
            plt.tight_layout()
            plt.show()


def save_graph_crop_as_png(graph: DetectedGraph, output_path: Path) -> None:
    """Save a detected graph crop as a PNG file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), graph.crop_image)
    logger.info(f"Saved graph crop: {output_path}")


# ============================================================================
# BOUNDING BOX EDITOR (INTERACTIVE MATPLOTLIB)
# ============================================================================

class BoundingBoxEditor:
    """
    Interactive matplotlib interface for editing bounding boxes.
    Allows users to drag and adjust crop boundaries in the viewer.
    """

    def __init__(self, image: np.ndarray, initial_bbox: Tuple[int, int, int, int]):
        """
        Initialize editor with full page image and initial bounding box.

        Args:
            image: Full page image (BGR)
            initial_bbox: Initial bounding box (x, y, width, height)
        """
        self.image = image
        self.bbox = list(initial_bbox)  # [x, y, w, h]
        self.edited = False
        self.fig = None
        self.ax = None
        self.rect_patch = None
        self.cid_press = None
        self.cid_release = None
        self.cid_motion = None
        self.dragging = False
        self.drag_corner = None

    def edit(self) -> Tuple[int, int, int, int]:
        """
        Launch interactive editor and return edited bounding box.

        Returns:
            Edited bounding box (x, y, width, height)
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except ImportError:
            logger.warning("matplotlib not available; skipping bounding box editor")
            return tuple(self.bbox)

        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.ax.imshow(rgb_image)
        self.ax.set_title("Bounding Box Editor - Drag corners/edges to adjust | Close to confirm")

        # Draw initial rectangle
        x, y, w, h = self.bbox
        self.rect_patch = Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
        self.ax.add_patch(self.rect_patch)

        # Connect mouse events
        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        plt.tight_layout()
        plt.show()

        return tuple(self.bbox)

    def _on_press(self, event):
        """Handle mouse button press."""
        if event.xdata is None or event.ydata is None:
            return

        x, y, w, h = self.bbox
        px, py = event.xdata, event.ydata

        # Detect which corner/edge is being dragged
        tolerance = 15
        if abs(px - (x + w)) < tolerance and abs(py - (y + h)) < tolerance:
            self.drag_corner = "bottom_right"
        elif abs(px - x) < tolerance and abs(py - y) < tolerance:
            self.drag_corner = "top_left"
        elif abs(px - (x + w)) < tolerance and abs(py - y) < tolerance:
            self.drag_corner = "top_right"
        elif abs(px - x) < tolerance and abs(py - (y + h)) < tolerance:
            self.drag_corner = "bottom_left"
        elif abs(px - (x + w)) < tolerance:
            self.drag_corner = "right"
        elif abs(px - x) < tolerance:
            self.drag_corner = "left"
        elif abs(py - (y + h)) < tolerance:
            self.drag_corner = "bottom"
        elif abs(py - y) < tolerance:
            self.drag_corner = "top"

        if self.drag_corner:
            self.dragging = True

    def _on_release(self, event):
        """Handle mouse button release."""
        self.dragging = False
        self.drag_corner = None

    def _on_motion(self, event):
        """Handle mouse motion to update bounding box."""
        if not self.dragging or event.xdata is None or event.ydata is None:
            return

        x, y, w, h = self.bbox
        px, py = int(event.xdata), int(event.ydata)
        img_h, img_w = self.image.shape[:2]

        # Update bounding box based on dragged corner/edge
        if self.drag_corner == "bottom_right":
            w = max(50, px - x)
            h = max(50, py - y)
        elif self.drag_corner == "top_left":
            new_x = min(px, x + w - 50)
            new_y = min(py, y + h - 50)
            w += x - new_x
            h += y - new_y
            x, y = new_x, new_y
        elif self.drag_corner == "top_right":
            w = max(50, px - x)
            new_y = min(py, y + h - 50)
            h += y - new_y
            y = new_y
        elif self.drag_corner == "bottom_left":
            new_x = min(px, x + w - 50)
            h = max(50, py - y)
            w += x - new_x
            x = new_x
        elif self.drag_corner == "right":
            w = max(50, px - x)
        elif self.drag_corner == "left":
            new_x = min(px, x + w - 50)
            w += x - new_x
            x = new_x
        elif self.drag_corner == "bottom":
            h = max(50, py - y)
        elif self.drag_corner == "top":
            new_y = min(py, y + h - 50)
            h += y - new_y
            y = new_y

        # Clamp to image bounds
        x = max(0, min(x, img_w - 50))
        y = max(0, min(y, img_h - 50))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        self.bbox = [x, y, w, h]

        # Update rectangle patch
        self.rect_patch.set_xy((x, y))
        self.rect_patch.set_width(w)
        self.rect_patch.set_height(h)
        self.fig.canvas.draw_idle()


# ============================================================================
# INTERACTIVE VALIDATION MODULE
# ============================================================================

class InteractiveValidator:
    """
    Provides human-in-the-loop validation of detected graph crops with
    interactive bounding box editing capability.
    """

    def __init__(self, enable_visualization: bool = False, full_page_image: Optional[np.ndarray] = None):
        """
        Initialize the validator.

        Args:
            enable_visualization: If True, display crops before asking for confirmation
            full_page_image: Full page image for bounding box editing (BGR)
        """
        self.enable_visualization = enable_visualization
        self.detector = GraphDetector()
        self.full_page_image = full_page_image

    def validate_crops(self, graphs: List[DetectedGraph]) -> List[DetectedGraph]:
        """
        Prompt user to confirm, skip, edit, or delete each detected graph crop.

        Args:
            graphs: List of DetectedGraph objects

        Returns:
            Filtered list of accepted (and possibly edited) graphs
        """
        if not graphs:
            logger.info("No graphs detected to validate.")
            return []

        accepted_graphs = []

        for graph in graphs:
            if self.enable_visualization:
                self._display_crop(graph)

            while True:
                user_input = input(
                    f"\nGraph #{graph.index + 1} (confidence: {graph.confidence:.2f}): "
                    f"[Enter] confirm | [e] edit bbox | [s] skip | [d] delete | [q] quit: "
                ).strip().lower()

                if user_input == "":
                    # Confirm
                    accepted_graphs.append(graph)
                    logger.info(f"Accepted graph #{graph.index + 1}")
                    break
                elif user_input == "e":
                    # Edit bounding box
                    if self.full_page_image is not None:
                        logger.info("Opening bounding box editor...")
                        editor = BoundingBoxEditor(self.full_page_image, graph.bbox)
                        new_bbox = editor.edit()
                        
                        # Re-crop image with new bbox
                        x, y, w, h = new_bbox
                        new_crop = self.full_page_image[y : y + h, x : x + w]
                        graph.bbox = new_bbox
                        graph.crop_image = new_crop
                        
                        logger.info(f"Updated bbox for graph #{graph.index + 1}: {new_bbox}")
                        # Show updated crop
                        if self.enable_visualization:
                            self._display_crop(graph)
                    else:
                        print("Full page image not available for bounding box editing.")
                        continue
                elif user_input == "s":
                    # Skip
                    logger.info(f"Skipped graph #{graph.index + 1}")
                    break
                elif user_input == "d":
                    # Delete
                    logger.info(f"Deleted graph #{graph.index + 1}")
                    break
                elif user_input == "q":
                    # Quit
                    logger.info("User quit validation.")
                    return accepted_graphs
                else:
                    print("Invalid input. Try: [Enter], [e], [s], [d], or [q]")

        return accepted_graphs

    @staticmethod
    def _display_crop(graph: DetectedGraph) -> None:
        """Display a single graph crop in matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping visualization")
            return

        rgb_crop = cv2.cvtColor(graph.crop_image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 6))
        plt.imshow(rgb_crop)
        plt.title(f"Graph #{graph.index + 1}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# ============================================================================
# IMAGE ENCODING & FILE HANDLING
# ============================================================================

def encode_image_to_base64(image_path: Path | np.ndarray) -> str:
    """
    Convert a PNG image file or numpy array to a base64-encoded string.

    Args:
        image_path: Path to image file or numpy array

    Returns:
        Base64-encoded string
    """
    if isinstance(image_path, np.ndarray):
        # Convert numpy array (BGR) to PNG bytes
        success, buffer = cv2.imencode(".png", image_path)
        if not success:
            raise ValueError("Failed to encode numpy array to PNG")
        image_bytes = buffer.tobytes()
    else:
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        with open(image_path, "rb") as f:
            image_bytes = f.read()

    return base64.standard_b64encode(image_bytes).decode("utf-8")


def get_sorted_png_files(processed_dir: Path) -> List[Path]:
    """
    Retrieve PNG files from processed_dir, sorted by page number.
    """
    if not processed_dir.exists() or not processed_dir.is_dir():
        logger.warning(f"Processed directory does not exist: {processed_dir}")
        return []

    png_files = sorted(processed_dir.glob("*.png"))

    def extract_page_number(filename: str) -> Tuple[int, str]:
        """Extract numeric page indicator from filename."""
        match = re.search(r"(\d+)", filename)
        if match:
            return (int(match.group(1)), filename)
        return (float("inf"), filename)

    png_files.sort(key=lambda p: extract_page_number(p.name))
    return png_files


# ============================================================================
# OPENAI MULTIMODAL STRUCTURED EXTRACTION
# ============================================================================

def extract_page_context(
    client: OpenAI,
    image_path: Path,
    model: str = "gpt-4o",
) -> PageContext:
    """
    Extract overall page context (metadata, methodology, chemical species).
    This is the first phase of analysis before graph-specific extraction.

    Args:
        client: Initialized OpenAI client
        image_path: Path to the full page PNG
        model: Model identifier (default: gpt-4o)

    Returns:
        PageContext object with general page information
    """
    logger.info(f"Extracting page context from {image_path.name}...")

    image_base64 = encode_image_to_base64(image_path)

    system_prompt = """You are an expert spectroscopist analyzing legacy scientific publications.
Extract the overall page context without focusing on individual graphs."""

    user_prompt = """Please extract the following page-level information:
- Document title, author, year, and publication details
- Sample description and experimental conditions
- Spectroscopic technique used
- List of chemical species or compounds studied
- Key methodology notes
- Overall findings or conclusions from the page

Provide factually grounded information only."""

    try:
        response = client.chat.completions.create(
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
            temperature=0.3,
            max_tokens=1500,
        )

        content = response.choices[0].message.content or ""

        # Parse response (simplified; in production, use regex or structured extraction)
        page_ctx = PageContext(
            title=extract_field(content, "title") or "Unknown Title",
            author=extract_field(content, "author") or "Unknown Author",
            year=int(extract_field(content, "year") or "2000"),
            sample_description=extract_field(content, "sample") or content[:500],
            experimental_technique=extract_field(content, "technique") or "Spectroscopy",
            chemical_species=extract_list_field(content, "species", "compound"),
            methodology_notes=extract_field(content, "methodology") or "",
            overall_findings=content[-500:],
        )

        logger.info(f"Extracted page context for '{page_ctx.title}'")
        return page_ctx

    except APIError as e:
        logger.error(f"OpenAI API error while extracting page context: {e}")
        raise


def extract_graph_analysis(
    client: OpenAI,
    graph_crop: np.ndarray,
    graph_index: int,
    page_context: PageContext,
    model: str = "gpt-4o",
) -> GraphAnalysis:
    """
    Extract spectroscopic analysis from a single detected graph.
    Uses the page context to provide targeted analysis.

    Args:
        client: Initialized OpenAI client
        graph_crop: Cropped image (numpy array) containing the graph
        graph_index: Sequential index of this graph (1-indexed)
        page_context: PageContext from the full page
        model: Model identifier (default: gpt-4o)

    Returns:
        GraphAnalysis object with peaks and graph description
    """
    logger.info(f"Extracting analysis for graph #{graph_index}...")

    graph_base64 = encode_image_to_base64(graph_crop)

    system_prompt = f"""You are an expert spectroscopist analyzing individual spectroscopic plots.
This graph is from a page about: {page_context.title}
Technique: {page_context.experimental_technique}
Sample: {page_context.sample_description}

Analyze this specific graph in detail."""

    user_prompt = f"""Please analyze this spectroscopic graph and extract:
- Graph title or label if visible
- Detected axis labels (X and Y axes)
- All visible spectroscopic peaks with:
  * Chemical assignment or functional group
  * Position (in eV, wavenumber, or other units)
  * Fitting bounds for curve fitting
  * Intensity (strong/medium/weak)
  * Special features (doublet, shoulder, etc.)
- Overall description of what this graph shows
- Any notable patterns or anomalies

Context: This is graph #{graph_index} from a page analyzing {', '.join(page_context.chemical_species) if page_context.chemical_species else 'unspecified compounds'}."""

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
                                "url": f"data:image/png;base64,{graph_base64}",
                            },
                        },
                    ],
                },
            ],
            response_format=GraphAnalysis,
            temperature=0.3,
            max_tokens=2000,
        )

        analysis = response.choices[0].message.parsed
        if analysis is None:
            raise ValueError("OpenAI returned null parsed response")

        analysis.graph_index = graph_index
        logger.info(
            f"Successfully extracted graph #{graph_index} "
            f"({len(analysis.detected_peaks)} peaks, confidence: {analysis.extraction_confidence:.2f})"
        )
        return analysis

    except APIError as e:
        logger.error(f"OpenAI API error while extracting graph #{graph_index}: {e}")
        raise


# ============================================================================
# HELPER FUNCTIONS FOR RESPONSE PARSING
# ============================================================================

def extract_field(text: str, field_name: str) -> Optional[str]:
    """Extract a single field value from text response."""
    patterns = [
        rf"{field_name}\s*:\s*([^\n]+)",
        rf"{field_name}\s*=\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_list_field(text: str, *field_names: str) -> List[str]:
    """Extract a list of values from text response."""
    for field_name in field_names:
        patterns = [
            rf"{field_name}[s]?\s*:\s*([^\n]+)",
            rf"{field_name}[s]?\s*=\s*([^\n]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                values = match.group(1).split(",")
                return [v.strip() for v in values if v.strip()]
    return []


# ============================================================================
# STORAGE & OBSIDIAN VAULT GENERATION
# ============================================================================

class PageStorageManager:
    """
    Manages the hierarchical storage structure: page → graphs → outputs.
    """

    def __init__(self, base_output_dir: Path):
        """Initialize storage manager with base output directory."""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def create_page_directory(self, page_name: str) -> Path:
        """Create a directory for a specific page."""
        page_dir = self.base_output_dir / page_name
        page_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created page directory: {page_dir}")
        return page_dir

    def save_page_summary(
        self,
        page_dir: Path,
        page_summary: PageAnalysisSummary,
    ) -> Path:
        """Save page-level summary as JSON."""
        summary_path = page_dir / "page_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(page_summary.model_dump(), f, indent=2)
        logger.info(f"Saved page summary: {summary_path}")
        return summary_path

    def save_graph_analysis(
        self,
        page_dir: Path,
        graph_analysis: GraphAnalysis,
    ) -> Path:
        """Save individual graph analysis as JSON."""
        graph_file = page_dir / f"graph_{graph_analysis.graph_index:02d}_analysis.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_analysis.model_dump(), f, indent=2)
        logger.info(f"Saved graph analysis: {graph_file}")
        return graph_file

    def save_graph_crop(
        self,
        page_dir: Path,
        graph: DetectedGraph,
    ) -> Path:
        """Save a graph crop as PNG for reference."""
        crop_file = page_dir / f"graph_{graph.index:02d}_crop.png"
        save_graph_crop_as_png(graph, crop_file)
        return crop_file


class ObsidianRefactoringAgent:
    """
    Transforms graph-level analysis into beautifully factorized Obsidian notes.
    """

    def __init__(self, vault_dir: Path):
        """Initialize with target vault directory."""
        self.vault_dir = Path(vault_dir)
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    def create_page_note(
        self,
        page_summary: PageAnalysisSummary,
        graph_analyses: List[GraphAnalysis],
    ) -> Path:
        """Generate a comprehensive Obsidian note for the entire page."""
        # Generate frontmatter
        frontmatter = self._generate_frontmatter(page_summary, graph_analyses)

        # Generate content sections
        content = f"# {page_summary.document_metadata.title}\n\n"
        content += self._generate_metadata_section(page_summary)
        content += self._generate_sample_section(page_summary)
        content += self._generate_graphs_overview(graph_analyses)

        full_content = frontmatter + content

        # Create filename and save
        safe_title = (
            page_summary.document_metadata.title.lower()
            .replace(" ", "-")
            .replace("/", "-")
            .replace(":", "")[:50]
        )
        filename = f"{safe_title}_{page_summary.document_metadata.page_source}.md"
        output_path = self.vault_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        logger.info(f"Obsidian note created: {output_path}")
        return output_path

    @staticmethod
    def _generate_frontmatter(
        page_summary: PageAnalysisSummary,
        graph_analyses: List[GraphAnalysis],
    ) -> str:
        """Generate YAML frontmatter."""
        frontmatter = "---\n"
        frontmatter += f"title: {page_summary.document_metadata.title}\n"

        tags = ["spectroscopy", page_summary.experimental_technique.lower().replace(" ", "-")]
        for species in page_summary.chemical_species[:3]:
            tags.append(species.lower().replace(" ", "-").replace("/", ""))
        frontmatter += f"tags: [{', '.join(tags)}]\n"

        frontmatter += f"source: {page_summary.document_metadata.author}, {page_summary.document_metadata.year}\n"
        frontmatter += f"page: {page_summary.document_metadata.page_source}\n"
        frontmatter += f"graphs_analyzed: {len(graph_analyses)}\n"
        frontmatter += f"extraction_confidence: {page_summary.page_extraction_confidence:.2f}\n"
        frontmatter += f"extracted: {datetime.now().isoformat()}\n"
        frontmatter += "---\n\n"

        return frontmatter

    @staticmethod
    def _generate_metadata_section(page_summary: PageAnalysisSummary) -> str:
        """Generate metadata section."""
        section = "## Document Metadata\n\n"
        section += f"**Title:** {page_summary.document_metadata.title}\n"
        section += f"**Authors:** {page_summary.document_metadata.author}\n"
        section += f"**Year:** {page_summary.document_metadata.year}\n"
        section += f"**Page:** {page_summary.document_metadata.page_source}\n\n"
        return section

    @staticmethod
    def _generate_sample_section(page_summary: PageAnalysisSummary) -> str:
        """Generate sample & methodology section."""
        section = "## Sample & Methodology\n\n"
        section += f"**Sample:** {page_summary.sample_description}\n\n"
        section += f"**Technique:** [[{page_summary.experimental_technique} Spectroscopy]]\n\n"
        if page_summary.methodology_notes:
            section += f"**Methodology:** {page_summary.methodology_notes}\n\n"
        section += "### Chemical Species Identified\n\n"
        for species in page_summary.chemical_species:
            section += f"- [[{species}]]\n"
        section += "\n"
        return section

    @staticmethod
    def _generate_graphs_overview(graph_analyses: List[GraphAnalysis]) -> str:
        """Generate overview of all graphs."""
        section = "## Spectroscopic Graphs Analyzed\n\n"
        section += f"**Total Graphs:** {len(graph_analyses)}\n\n"

        for graph in graph_analyses:
            section += f"### Graph {graph.graph_index}: "
            section += f"{graph.graph_title or 'Untitled'}\n\n"
            section += f"{graph.graph_description}\n\n"

            if graph.detected_peaks:
                section += "**Detected Peaks:**\n\n"
                section += "| Assignment | Position | Lower Bound | Upper Bound | Intensity |\n"
                section += "|---|---|---|---|---|\n"
                for peak in graph.detected_peaks:
                    lower, upper = peak.fitting_bounds
                    intensity = peak.intensity_estimate or "N/A"
                    section += (
                        f"| {peak.chemical_assignment} | {peak.estimated_position_ev:.2f} "
                        f"| {lower:.2f} | {upper:.2f} | {intensity} |\n"
                    )
                section += "\n"

            section += f"*Confidence: {graph.extraction_confidence:.2%}*\n\n"

        return section


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def process_page_with_graphs(
    image_path: Path,
    client: OpenAI,
    output_dir: Path,
    vault_dir: Path,
    validate_crops: bool = False,
    enable_visualization: bool = False,
) -> Dict[str, Any]:
    """
    Process a single page: extract context, detect graphs, analyze each graph.

    Args:
        image_path: Path to the page PNG
        client: Initialized OpenAI client
        output_dir: Base output directory
        vault_dir: Obsidian vault directory
        validate_crops: Enable interactive validation
        enable_visualization: Enable matplotlib visualization

    Returns:
        Dictionary with processing results
    """
    results = {
        "image": image_path.name,
        "success": False,
        "page_context": None,
        "graphs_detected": 0,
        "graphs_analyzed": 0,
        "errors": [],
    }

    try:
        # Phase 1: Extract page context
        page_context = extract_page_context(client, image_path)

        # Phase 2: Detect graphs
        detector = GraphDetector()
        detected_graphs = detector.detect_graphs(image_path)
        results["graphs_detected"] = len(detected_graphs)

        if not detected_graphs:
            logger.warning(f"No graphs detected in {image_path.name}. Treating page as single entity.")
            detected_graphs = []

        # Phase 3: Interactive validation (optional)
        if validate_crops and detected_graphs:
            full_page_image = cv2.imread(str(image_path))
            validator = InteractiveValidator(
                enable_visualization=enable_visualization,
                full_page_image=full_page_image,
            )
            detected_graphs = validator.validate_crops(detected_graphs)
            logger.info(f"Validated {len(detected_graphs)} graph(s) after user input")

        # Phase 4: Create page directory and storage manager
        storage_manager = PageStorageManager(output_dir)
        safe_page_name = (
            image_path.stem.lower().replace(" ", "_").replace(".", "")[:80]
        )
        page_dir = storage_manager.create_page_directory(safe_page_name)

        # Phase 5: Analyze each graph
        graph_analyses = []
        for graph in detected_graphs:
            try:
                analysis = extract_graph_analysis(
                    client,
                    graph.crop_image,
                    graph.index + 1,
                    page_context,
                )
                graph_analyses.append(analysis)
                storage_manager.save_graph_analysis(page_dir, analysis)
                storage_manager.save_graph_crop(page_dir, graph)
                results["graphs_analyzed"] += 1
            except Exception as e:
                error_msg = f"Error analyzing graph #{graph.index + 1}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        # Phase 6: Save page summary
        page_summary = PageAnalysisSummary(
            document_metadata=DocumentMetadata(
                title=page_context.title,
                author=page_context.author,
                year=page_context.year,
                page_source=image_path.stem,
            ),
            sample_description=page_context.sample_description,
            experimental_technique=page_context.experimental_technique,
            chemical_species=page_context.chemical_species,
            methodology_notes=page_context.methodology_notes,
            overall_findings=page_context.overall_findings,
            page_extraction_confidence=0.9,
        )
        storage_manager.save_page_summary(page_dir, page_summary)

        # Phase 7: Generate Obsidian note
        vault_agent = ObsidianRefactoringAgent(vault_dir)
        note_path = vault_agent.create_page_note(page_summary, graph_analyses)

        results["success"] = True
        results["page_context"] = page_context.__dict__
        results["output_dir"] = str(page_dir)
        results["obsidian_note"] = str(note_path)

    except Exception as e:
        error_msg = f"Error processing {image_path.name}: {str(e)}"
        logger.error(error_msg)
        results["errors"].append(error_msg)

    return results


def process_image_batch(
    processed_dir: Path = Path("Deposit/processed"),
    output_dir: Path = Path("ExtractionOutput"),
    vault_dir: Path = Path("ObsidianVault"),
    validate_crops: bool = False,
    enable_visualization: bool = False,
) -> Dict[str, Any]:
    """
    Process all PNG images with graph detection and analysis.

    Args:
        processed_dir: Directory containing processed PNG files
        output_dir: Base output directory for hierarchical storage
        vault_dir: Output directory for Obsidian markdown files
        validate_crops: Enable interactive validation of detected crops
        enable_visualization: Enable matplotlib visualization

    Returns:
        Summary dictionary of processing results
    """
    api_key = load_environment()
    client = OpenAI(api_key=api_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    vault_dir.mkdir(parents=True, exist_ok=True)

    png_files = get_sorted_png_files(processed_dir)
    if not png_files:
        logger.warning(f"No PNG files found in {processed_dir}")
        return {
            "status": "no_files",
            "processed_count": 0,
            "errors": ["No PNG files found in processed directory"],
        }

    logger.info(f"Found {len(png_files)} PNG file(s) to process")

    results = {
        "status": "success",
        "total_processed": len(png_files),
        "successful": 0,
        "failed": 0,
        "total_graphs_detected": 0,
        "total_graphs_analyzed": 0,
        "page_results": [],
        "errors": [],
    }

    for image_path in png_files:
        page_result = process_page_with_graphs(
            image_path,
            client,
            output_dir,
            vault_dir,
            validate_crops=validate_crops,
            enable_visualization=enable_visualization,
        )

        results["page_results"].append(page_result)
        if page_result["success"]:
            results["successful"] += 1
            results["total_graphs_detected"] += page_result["graphs_detected"]
            results["total_graphs_analyzed"] += page_result["graphs_analyzed"]
        else:
            results["failed"] += 1
            results["errors"].extend(page_result["errors"])

    # Summary logging
    logger.info("\n" + "=" * 70)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total pages processed: {results['successful']}/{results['total_processed']}")
    logger.info(f"Total graphs detected: {results['total_graphs_detected']}")
    logger.info(f"Total graphs analyzed: {results['total_graphs_analyzed']}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Obsidian vault: {vault_dir}")

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Graph-centric spectroscopy analysis: detect multiple graphs per page, "
            "perform targeted analysis, generate hierarchical outputs."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="Deposit/processed",
        help="Directory containing processed PNG files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ExtractionOutput",
        help="Base output directory for hierarchical page/graph storage.",
    )
    parser.add_argument(
        "--vault-dir",
        type=str,
        default="ObsidianVault",
        help="Output directory for Obsidian markdown files.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Enable interactive validation of detected graph crops.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable matplotlib visualization during validation.",
    )

    args = parser.parse_args()

    summary = process_image_batch(
        processed_dir=Path(args.processed_dir),
        output_dir=Path(args.output_dir),
        vault_dir=Path(args.vault_dir),
        validate_crops=args.validate,
        enable_visualization=args.visualize,
    )

    sys.exit(0 if summary["status"] == "success" else 1)
