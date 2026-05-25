# chart_feature_detector_v2.py
"""
Chart Feature Detection Agent - ROBUST VERSION
Fixed OpenCV array mismatch errors and edge cases
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from scipy import ndimage
from scipy.signal import find_peaks
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MarkerPoint:
    """Represents a detected marker/peak in the chart"""
    x: float
    y: float
    marker_type: str
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
    axis_type: str
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
# ROBUST CHART MARKER DETECTOR
# ============================================================================

class ChartMarkerDetector:
    """
    Detects visual markers and peaks in spectroscopy charts.
    FIXED: Handles OpenCV array dimension mismatches.
    """

    def __init__(self, image_path: str, debug: bool = False):
        """Initialize with robust error handling"""
        self.image_path = Path(image_path)
        self.debug = debug
        self.detected_markers: List[MarkerPoint] = []
        self.chart_features: Optional[ChartFeatures] = None

        # Load image with validation
        self._load_image_safe()

    def _load_image_safe(self):
        """Safely load image and handle format conversions"""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Chart image not found: {self.image_path}")

        self.original_image = cv2.imread(str(self.image_path))
        if self.original_image is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        logger.info(f"✓ Loaded image: {self.image_path.name} "
                    f"(shape: {self.original_image.shape})")

        # Get dimensions
        self.height, self.width = self.original_image.shape[:2]

        # Safe color conversions (CRITICAL FIX)
        try:
            self.image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.image_hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            self.image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            logger.error(f"Color conversion error: {e}")
            # Fallback: use original if conversion fails
            self.image_rgb = self.original_image.copy()
            self.image_hsv = self.original_image.copy()
            self.image_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

    def detect_markers_by_color(self,
                                color_name: str,
                                color_range: Dict[str, Tuple],
                                marker_type: str = 'circle') -> List[MarkerPoint]:
        """
        Detect markers by color range in HSV space.
        FIXED: Proper array dimension handling
        """
        logger.info(f"🔍 Detecting {color_name} {marker_type}s...")

        try:
            lower = np.array(color_range['lower'], dtype=np.uint8)
            upper = np.array(color_range['upper'], dtype=np.uint8)

            # Verify HSV image format
            if self.image_hsv.shape[2] != 3:
                logger.warning(f"HSV image has {self.image_hsv.shape[2]} channels, expected 3")
                return []

            # Create mask (FIXED: proper array dimensions)
            mask = cv2.inRange(self.image_hsv, lower, upper)

            # Verify mask dimensions match
            if mask.shape != self.image_hsv.shape[:2]:
                logger.error("Mask dimensions don't match image")
                return []

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

            markers = []
            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by size
                if area < 10 or area > 10000:
                    continue

                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                # Circularity as confidence
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    confidence = min(max(circularity, 0.0), 1.0)
                else:
                    confidence = 0.5

                # Normalize coordinates
                norm_x = center_x / self.width
                norm_y = center_y / self.height

                # Clip to valid range
                norm_x = max(0.0, min(norm_x, 1.0))
                norm_y = max(0.0, min(norm_y, 1.0))

                # Get color at marker (FIXED: safer extraction)
                try:
                    roi = self.original_image[max(0, y):min(self.height, y+h),
                                             max(0, x):min(self.width, x+w)]
                    if roi.size > 0:
                        color_bgr = cv2.mean(roi)[:3]
                        color_rgb = (int(color_bgr[2]), int(color_bgr[1]),
                                    int(color_bgr[0]))
                    else:
                        color_rgb = (0, 0, 0)
                except Exception as e:
                    logger.warning(f"Color extraction failed: {e}")
                    color_rgb = (0, 0, 0)

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

            logger.info(f"  ✓ Found {len(markers)} {color_name} markers")
            self.detected_markers.extend(markers)
            return markers

        except cv2.error as e:
            logger.error(f"OpenCV error in color detection: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in color detection: {e}")
            return []

    def detect_peaks_by_intensity(self) -> List[MarkerPoint]:
        """
        Detect peaks in grayscale intensity profile.
        FIXED: Dimension checks and edge case handling
        """
        logger.info("📈 Detecting peaks by intensity...")

        try:
            if self.image_gray.size == 0:
                logger.warning("Grayscale image is empty")
                return []

            # Compute profiles - FIXED: proper axis specification
            horizontal_profile = np.mean(self.image_gray, axis=0)
            vertical_profile = np.mean(self.image_gray, axis=1)

            # Verify profile shapes
            if horizontal_profile.shape[0] != self.width:
                logger.error("Horizontal profile dimension mismatch")
                horizontal_profile = np.linspace(0, 255, self.width)

            if vertical_profile.shape[0] != self.height:
                logger.error("Vertical profile dimension mismatch")
                vertical_profile = np.linspace(0, 255, self.height)

            # Normalize
            h_max = np.max(horizontal_profile)
            v_max = np.max(vertical_profile)

            horizontal_profile = horizontal_profile / (h_max + 1e-6)
            vertical_profile = vertical_profile / (v_max + 1e-6)

            # Find peaks with safe parameters
            min_distance = max(5, self.width // 50)

            try:
                peaks_x, _ = find_peaks(horizontal_profile, height=0.3,
                                       distance=min_distance)
                peaks_y, _ = find_peaks(vertical_profile, height=0.3,
                                       distance=min_distance)
            except Exception as e:
                logger.warning(f"Peak finding failed: {e}")
                return []

            markers = []

            # Create 2D grid of peaks (FIXED: avoid redundant pairs)
            for px in peaks_x[:5]:  # Limit to top peaks
                for py in peaks_y[:5]:
                    if px >= self.width or py >= self.height:
                        continue

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

            logger.info(f"  ✓ Found {len(markers)} intensity peaks")
            self.detected_markers.extend(markers)
            return markers

        except Exception as e:
            logger.error(f"Error in peak detection: {e}")
            return []

    def detect_stars_and_symbols(self) -> List[MarkerPoint]:
        """
        Detect star markers and common plot symbols.
        FIXED: Robust contour handling
        """
        logger.info("⭐ Detecting star symbols...")

        try:
            # Adaptive threshold
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

                if area < 20 or area > 5000:
                    continue

                # Fit ellipse if possible (FIXED: requires at least 5 points)
                if len(contour) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        ((cx, cy), (w, h), angle) = ellipse

                        # Aspect ratio check
                        aspect_ratio = w / (h + 1e-6)
                        if not (0.5 < aspect_ratio < 2.0):
                            continue

                        # Coordinate validation
                        if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
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
                    except cv2.error as e:
                        logger.debug(f"Ellipse fitting failed: {e}")
                        continue

            logger.info(f"  ✓ Found {len(markers)} star symbols")
            self.detected_markers.extend(markers)
            return markers

        except Exception as e:
            logger.error(f"Error in symbol detection: {e}")
            return []

    def extract_axis_labels(self) -> Tuple[AxisInfo, AxisInfo]:
        """
        Extract axis labels and ranges from chart.
        Placeholder for OCR integration.
        """
        logger.info("📊 Extracting axis information...")

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
        Master detection function - runs all methods safely.
        """
        logger.info("\n" + "="*60)
        logger.info("🎯 STARTING CHART FEATURE DETECTION")
        logger.info("="*60 + "\n")

        # Define HSV color ranges
        colors = {
            'red': {'lower': (0, 100, 100), 'upper': (10, 255, 255)},
            'blue': {'lower': (100, 100, 100), 'upper': (130, 255, 255)},
            'green': {'lower': (40, 50, 50), 'upper': (80, 255, 255)},
        }

        # Run all detection methods
        for color_name, color_range in colors.items():
            self.detect_markers_by_color(color_name, color_range)

        self.detect_peaks_by_intensity()
        self.detect_stars_and_symbols()

        x_axis, y_axis = self.extract_axis_labels()

        # Calculate overall confidence
        if self.detected_markers:
            confidence = np.mean([m.confidence for m in self.detected_markers])
        else:
            confidence = 0.0

        self.chart_features = ChartFeatures(
            title="Detected Chart",
            markers=self.detected_markers,
            x_axis=x_axis,
            y_axis=y_axis,
            legend_items=[],
            grid_visible=True,
            background_color=(255, 255, 255),
            extraction_confidence=confidence
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"✓ DETECTION COMPLETE")
        logger.info(f"  Total markers detected: {len(self.detected_markers)}")
        logger.info(f"  Overall confidence: {confidence:.2%}")
        logger.info(f"{'='*60}\n")

        return self.chart_features


# ============================================================================
# DATA RECONSTRUCTION
# ============================================================================

class ChartDataReconstructor:
    """Converts detected features into chart data (X,Y coordinates)"""

    def __init__(self, chart_features: ChartFeatures):
        self.features = chart_features

    def denormalize_coordinates(self, marker: MarkerPoint) -> Tuple[float, float]:
        """Convert normalized (0-1) to actual data coordinates"""
        x_axis = self.features.x_axis
        y_axis = self.features.y_axis

        data_x = x_axis.min_value + marker.x * (x_axis.max_value - x_axis.min_value)
        data_y = y_axis.min_value + (1.0 - marker.y) * (y_axis.max_value - y_axis.min_value)

        return data_x, data_y

    def generate_dataframe(self) -> pd.DataFrame:
        """Generate pandas DataFrame from detected markers"""
        data = []

        for point_id, marker in enumerate(self.features.markers, start=1):
            data_x, data_y = self.denormalize_coordinates(marker)

            data.append({
                'point_id': point_id,
                f'{self.features.x_axis.label}': data_x,
                f'{self.features.y_axis.label}': data_y,
                'marker_type': marker.marker_type,
                'confidence': marker.confidence,
            })

        df = pd.DataFrame(data)

        # Sort by x value
        x_col = f'{self.features.x_axis.label}'
        df = df.sort_values(x_col).reset_index(drop=True)

        return df

    def save_to_csv(self, filepath: str) -> str:
        """Save reconstructed data to CSV"""
        df = self.generate_dataframe()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"✓ CSV saved: {filepath}")
        return filepath

    def save_features_json(self, filepath: str) -> str:
        """Save all detected features to JSON"""
        output = {
            'title': self.features.title,
            'extraction_confidence': self.features.extraction_confidence,
            'marker_count': len(self.features.markers),
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
            'markers': [m.to_dict() for m in self.features.markers]
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"✓ Features JSON saved: {filepath}")
        return filepath


# ============================================================================
# PLOTLY VERIFICATION
# ============================================================================

def create_verification_plot(csv_path: str, output_html: str = None):
    """Create interactive Plotly visualization"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("⚠️  Plotly not installed. Install: pip install plotly")
        return

    df = pd.read_csv(csv_path)

    # Find data columns
    x_col = [c for c in df.columns if 'energy' in c.lower()][0] \
            if any('energy' in c.lower() for c in df.columns) else df.columns[1]
    y_col = [c for c in df.columns if 'intensity' in c.lower()][0] \
            if any('intensity' in c.lower() for c in df.columns) else df.columns[2]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers+lines',
        marker=dict(
            size=8,
            color=df['confidence'] if 'confidence' in df.columns else 'blue',
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Confidence")
        ),
        line=dict(width=2),
        name='Extracted Data'
    ))

    fig.update_layout(
        title="✓ Plotly Verification: Extracted Chart Data",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        height=600,
        showlegend=True
    )

    if output_html:
        fig.write_html(output_html)
        logger.info(f"✓ Verification HTML saved: {output_html}")

    try:
        fig.show()
    except Exception as e:
        logger.warning(f"Could not display plot: {e}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def process_chart_image(image_path: str, output_dir: str = 'chart_extraction/'):
    """
    Complete workflow: Detect → Extract → Verify
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Detect
        detector = ChartMarkerDetector(image_path, debug=False)
        features = detector.detect_all_features()

        # Step 2: Reconstruct
        reconstructor = ChartDataReconstructor(features)
        csv_path = reconstructor.save_to_csv(str(output_path / 'extracted_chart_data.csv'))
        json_path = reconstructor.save_features_json(str(output_path / 'detected_features.json'))

        # Step 3: Verify
        html_path = str(output_path / 'verification_plot.html')
        create_verification_plot(csv_path, html_path)

        return csv_path, json_path, html_path

    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        try:
            process_chart_image(image_path)
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            sys.exit(1)
    else:
        print("""
        Usage: python chart_feature_detector_v2.py <image_path> [output_dir]
        
        Example:
            python chart_feature_detector_v2.py training_data/graph_only.png
        """)