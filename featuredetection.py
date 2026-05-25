"""
feed vision model image
model employs feature detection (markers, peaks, axes, labels)
extracted features go in csv, and obsidian stuff
plotly verifies graph for accuracy check
"""

# chart_feature_detector.py
"""
Chart Feature Detection Agent
Detects markers, symbols, peaks, and other visual features in spectroscopy chart images.
Extracts chart data to CSV format that can be verified with Plotly.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from scipy import ndimage
from scipy.signal import find_peaks
import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================================
# DATA CLASSES FOR DETECTED FEATURES
# ============================================================================

@dataclass
class MarkerPoint:
    """Represents a detected marker/peak in the chart"""
    x: float
    y: float
    marker_type: str  # 'star', 'circle', 'square', 'cross', etc.
    color: Tuple[int, int, int]
    confidence: float
    size: int
    label: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'x': self.x,
            'y': self.y,
            'marker_type': self.marker_type,
            'color_rgb': self.color,
            'confidence': self.confidence,
            'size': self.size,
            'label': self.label
        }


@dataclass
class AxisInfo:
    """Extracted axis information from chart"""
    axis_type: str  # 'x' or 'y'
    label: str
    min_value: float
    max_value: float
    unit: str
    tick_positions: List[float]
    tick_labels: List[str]


@dataclass
class ChartFeatures:
    """Complete set of detected chart features"""
    title: str
    markers: List[MarkerPoint]
    x_axis: AxisInfo
    y_axis: AxisInfo
    legend_items: List[Dict]
    grid_visible: bool
    background_color: Tuple[int, int, int]
    extraction_confidence: float


# ============================================================================
# MARKER DETECTION ENGINE
# ============================================================================

class ChartMarkerDetector:
    """
    Detects visual markers and peaks in spectroscopy charts using
    image processing and computer vision techniques.
    """

    def __init__(self, image_path: str, debug: bool = False):
        """
        Initialize the chart marker detector.

        Args:
            image_path: Path to chart image (PNG/JPG)
            debug: Enable debug visualizations
        """
        self.image_path = Path(image_path)
        self.debug = debug

        # Load and validate image
        if not self.image_path.exists():
            raise FileNotFoundError(f"Chart image not found: {image_path}")

        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        self.height, self.width = self.original_image.shape[:2]
        self.image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.image_hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        self.image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        self.detected_markers: List[MarkerPoint] = []
        self.chart_features: Optional[ChartFeatures] = None

    def detect_markers_by_color(self,
                                color_range: Dict[str, Tuple],
                                marker_type: str = 'star') -> List[MarkerPoint]:
        """
        Detect markers by color range in HSV space.

        Args:
            color_range: Dictionary with 'lower' and 'upper' HSV bounds
            marker_type: Type of marker detected

        Returns:
            List of detected MarkerPoint objects
        """
        lower = np.array(color_range['lower'])
        upper = np.array(color_range['upper'])

        # Create mask for this color range
        mask = cv2.inRange(self.image_hsv, lower, upper)

        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        markers = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by size (avoid noise and image edges)
            if area < 10 or area > 10000:
                continue

            # Get contour properties
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            # Calculate circularity as confidence metric
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                confidence = min(circularity, 1.0)
            else:
                confidence = 0.5

            # Normalize coordinates to 0-1 range
            norm_x = center_x / self.width
            norm_y = center_y / self.height

            # Get average color at marker location
            color_bgr = cv2.mean(self.original_image, mask[y:y + h, x:x + w])[:3]
            color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))

            marker = MarkerPoint(
                x=norm_x,
                y=norm_y,
                marker_type=marker_type,
                color=color_rgb,
                confidence=confidence,
                size=max(w, h),
                label=None
            )
            markers.append(marker)

        self.detected_markers.extend(markers)
        return markers

    def detect_peaks_by_intensity(self) -> List[MarkerPoint]:
        """
        Detect peaks by analyzing horizontal intensity profile (for spectroscopy data).
        Works by finding local maxima in the grayscale image.

        Returns:
            List of detected peak positions
        """
        # Compute horizontal intensity profile
        horizontal_profile = np.mean(self.image_gray, axis=0)
        vertical_profile = np.mean(self.image_gray, axis=1)

        # Normalize
        horizontal_profile = horizontal_profile / (np.max(horizontal_profile) + 1e-6)
        vertical_profile = vertical_profile / (np.max(vertical_profile) + 1e-6)

        # Find peaks in profiles
        peaks_x, _ = find_peaks(horizontal_profile, height=0.3, distance=20)
        peaks_y, _ = find_peaks(vertical_profile, height=0.3, distance=20)

        markers = []

        # Convert peak indices back to image coordinates
        for px in peaks_x:
            for py in peaks_y:
                norm_x = px / self.width
                norm_y = py / self.height

                intensity = self.image_gray[py, px] / 255.0

                marker = MarkerPoint(
                    x=norm_x,
                    y=norm_y,
                    marker_type='peak',
                    color=(0, 0, 0),
                    confidence=intensity,
                    size=10,
                    label=None
                )
                markers.append(marker)

        self.detected_markers.extend(markers)
        return markers

    def detect_stars_and_symbols(self) -> List[MarkerPoint]:
        """
        Specialized detection for star markers (*) and common plot symbols.
        Uses template matching or shape analysis.

        Returns:
            List of star/symbol markers
        """
        # Convert to binary image with adaptive threshold
        binary = cv2.adaptiveThreshold(
            self.image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        markers = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Star markers typically have specific size ranges
            if area < 20 or area > 5000:
                continue

            # Fit ellipse if possible
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                ((cx, cy), (w, h), angle) = ellipse

                # Aspect ratio check (stars often have ~1:1 ratio)
                aspect_ratio = w / (h + 1e-6)
                if not (0.5 < aspect_ratio < 2.0):
                    continue

                norm_x = cx / self.width
                norm_y = cy / self.height

                marker = MarkerPoint(
                    x=norm_x,
                    y=norm_y,
                    marker_type='star',
                    color=(255, 0, 0),
                    confidence=0.8,
                    size=int(max(w, h)),
                    label=None
                )
                markers.append(marker)

        self.detected_markers.extend(markers)
        return markers

    def extract_axis_labels(self) -> Tuple[AxisInfo, AxisInfo]:
        """
        Extract axis labels and ranges from chart edges.
        This is a simplified version - real implementation would use OCR.

        Returns:
            Tuple of (x_axis, y_axis) AxisInfo objects
        """
        # Placeholder: In production, use Tesseract OCR or vision API
        x_axis = AxisInfo(
            axis_type='x',
            label='Binding Energy (eV)',
            min_value=0.0,
            max_value=100.0,
            unit='eV',
            tick_positions=[0, 25, 50, 75, 100],
            tick_labels=['0', '25', '50', '75', '100']
        )

        y_axis = AxisInfo(
            axis_type='y',
            label='Intensity (Counts)',
            min_value=0.0,
            max_value=150.0,
            unit='Counts',
            tick_positions=[0, 50, 100, 150],
            tick_labels=['0', '50', '100', '150']
        )

        return x_axis, y_axis

    def detect_all_features(self) -> ChartFeatures:
        """
        Master detection function that runs all feature detection methods.

        Returns:
            ChartFeatures object containing all detected elements
        """
        # Define color ranges for common plot marker colors (in HSV)
        red_range = {'lower': (0, 100, 100), 'upper': (10, 255, 255)}
        blue_range = {'lower': (100, 100, 100), 'upper': (130, 255, 255)}
        green_range = {'lower': (40, 50, 50), 'upper': (80, 255, 255)}

        print("🔍 Detecting markers by color...")
        self.detect_markers_by_color(red_range, marker_type='circle')
        self.detect_markers_by_color(blue_range, marker_type='square')
        self.detect_markers_by_color(green_range, marker_type='triangle')

        print("📈 Detecting peaks by intensity...")
        self.detect_peaks_by_intensity()

        print("⭐ Detecting star symbols...")
        self.detect_stars_and_symbols()

        print("📊 Extracting axis information...")
        x_axis, y_axis = self.extract_axis_labels()

        self.chart_features = ChartFeatures(
            title="Detected Chart",
            markers=self.detected_markers,
            x_axis=x_axis,
            y_axis=y_axis,
            legend_items=[],
            grid_visible=True,
            background_color=(255, 255, 255),
            extraction_confidence=np.mean([m.confidence for m in self.detected_markers])
            if self.detected_markers else 0.0
        )

        return self.chart_features


# ============================================================================
# CSV GENERATION & RECONSTRUCTION
# ============================================================================

class ChartDataReconstructor:
    """Converts detected features back into chart data (X,Y coordinates)"""

    def __init__(self, chart_features: ChartFeatures):
        """
        Initialize with detected chart features.

        Args:
            chart_features: ChartFeatures object from detector
        """
        self.features = chart_features

    def denormalize_coordinates(self, marker: MarkerPoint) -> Tuple[float, float]:
        """
        Convert normalized (0-1) coordinates to actual data coordinates
        using axis information.

        Args:
            marker: MarkerPoint with normalized coordinates

        Returns:
            Tuple of (data_x, data_y) in actual units
        """
        x_axis = self.features.x_axis
        y_axis = self.features.y_axis

        # Denormalize X
        data_x = x_axis.min_value + marker.x * (x_axis.max_value - x_axis.min_value)

        # Denormalize Y (flip because image coordinates are inverted)
        data_y = y_axis.min_value + (1.0 - marker.y) * (y_axis.max_value - y_axis.min_value)

        return data_x, data_y

    def generate_dataframe(self) -> pd.DataFrame:
        """
        Generate pandas DataFrame from detected markers.

        Returns:
            DataFrame with columns: point_id, x_value, y_value, marker_type, confidence
        """
        data = []

        for point_id, marker in enumerate(self.features.markers, start=1):
            data_x, data_y = self.denormalize_coordinates(marker)

            data.append({
                'point_id': point_id,
                f'{self.features.x_axis.label}': data_x,
                f'{self.features.y_axis.label}': data_y,
                'marker_type': marker.marker_type,
                'confidence': marker.confidence,
                'size_pixels': marker.size,
            })

        return pd.DataFrame(data)

    def save_to_csv(self, filepath: str) -> str:
        """
        Save reconstructed data to CSV file.

        Args:
            filepath: Output CSV path

        Returns:
            Path to saved CSV
        """
        df = self.generate_dataframe()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"✓ Chart data saved to {filepath}")
        return filepath

    def save_features_json(self, filepath: str) -> str:
        """
        Save all detected features to JSON for analysis.

        Args:
            filepath: Output JSON path

        Returns:
            Path to saved JSON
        """
        output = {
            'title': self.features.title,
            'extraction_confidence': self.features.extraction_confidence,
            'axes': {
                'x': {
                    'label': self.features.x_axis.label,
                    'min': self.features.x_axis.min_value,
                    'max': self.features.x_axis.max_value,
                    'unit': self.features.x_axis.unit,
                },
                'y': {
                    'label': self.features.y_axis.label,
                    'min': self.features.y_axis.min_value,
                    'max': self.features.y_axis.max_value,
                    'unit': self.features.y_axis.unit,
                }
            },
            'markers': [m.to_dict() for m in self.features.markers],
            'marker_count': len(self.features.markers),
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Features saved to {filepath}")
        return filepath


# ============================================================================
# PLOTLY VERIFICATION VISUALIZATION
# ============================================================================

def create_verification_plot(csv_path: str, output_html: str = None):
    """
    Create interactive Plotly visualization to verify extracted data.

    Args:
        csv_path: Path to extracted CSV file
        output_html: Optional path to save HTML visualization
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        print("⚠️  Plotly not installed. Install with: pip install plotly")
        return

    # Load extracted data
    df = pd.read_csv(csv_path)

    # Find data columns (skip metadata)
    x_col = [c for c in df.columns if 'energy' in c.lower() or c == 'x_value'][0] \
        if any('energy' in c.lower() or c == 'x_value' for c in df.columns) \
        else df.columns[1]

    y_col = [c for c in df.columns if 'intensity' in c.lower() or c == 'y_value'][0] \
        if any('intensity' in c.lower() or c == 'y_value' for c in df.columns) \
        else df.columns[2]

    # Create scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers+lines',
        marker=dict(
            size=df['size_pixels'] / 10 if 'size_pixels' in df.columns else 10,
            color=df['confidence'] if 'confidence' in df.columns else 'blue',
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence")
        ),
        line=dict(width=2),
        name='Extracted Data'
    ))

    fig.update_layout(
        title="Plotly Verification: Extracted Chart Data",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        height=600,
        showlegend=True
    )

    if output_html:
        fig.write_html(output_html)
        print(f"✓ Verification plot saved to {output_html}")

    fig.show()


# ============================================================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================================================

def process_chart_image(image_path: str, output_dir: str = 'chart_extraction/'):
    """
    Complete workflow: Detect features → Extract CSV → Generate Plotly verification

    Args:
        image_path: Path to chart image
        output_dir: Directory for outputs
    """
    print(f"\n{'=' * 60}")
    print(f"🎯 CHART FEATURE EXTRACTION WORKFLOW")
    print(f"{'=' * 60}\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Detect features
    print(f"📖 Processing chart image: {image_path}\n")
    detector = ChartMarkerDetector(image_path, debug=True)
    features = detector.detect_all_features()

    print(f"\n✓ Detected {len(features.markers)} markers/peaks")
    print(f"  Extraction confidence: {features.extraction_confidence:.2%}\n")

    # Step 2: Generate CSV
    print("📊 Reconstructing chart data...\n")
    reconstructor = ChartDataReconstructor(features)
    csv_path = reconstructor.save_to_csv(str(output_path / 'extracted_chart_data.csv'))

    # Step 3: Save features JSON
    json_path = reconstructor.save_features_json(str(output_path / 'detected_features.json'))

    # Step 4: Generate Plotly verification
    print("\n📈 Creating Plotly verification visualization...\n")
    html_path = str(output_path / 'verification_plot.html')
    create_verification_plot(csv_path, html_path)

    print(f"\n{'=' * 60}")
    print(f"✓ EXTRACTION COMPLETE")
    print(f"  CSV: {csv_path}")
    print(f"  Features JSON: {json_path}")
    print(f"  Verification HTML: {html_path}")
    print(f"{'=' * 60}\n")

    return csv_path, json_path, html_path


if __name__ == "__main__":
    # Example usage
    # First, you need a chart image to process
    # For testing, use: python chart_feature_detector.py path/to/your/chart.png

    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        process_chart_image(image_path)
    else:
        print("""
        Usage: python chart_feature_detector.py <image_path> [output_dir]

        Example:
            python chart_feature_detector.py training_data/graph_only.png
        """)