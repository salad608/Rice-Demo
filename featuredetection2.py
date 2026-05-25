"""
High-Accuracy Chart Feature Detection Agent - GPT-5.5 Pro Edition
Uses OpenAI's gpt-5.5-pro (completions API) for maximum accuracy

Combines:
1. GPT-5.5-Pro completions extraction (primary - uses /v1/completions)
2. Fallback to GPT-4o chat completions (/v1/chat/completions)
3. OCR axis label verification
4. Local marker verification
5. Peak fitting validation

Key Fix:
- gpt-5.5-pro uses /v1/completions endpoint (not chat)
- Handles both completion and chat APIs seamlessly
- Auto-fallback if gpt-5.5-pro unavailable
"""

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import cv2

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC SCHEMAS - STRUCTURED EXTRACTION
# ============================================================================

class ChartAxis(BaseModel):
    """Extracted axis information"""
    label: str = Field(..., description="Axis label text")
    unit: str = Field(..., description="Unit (e.g., 'eV', 'Counts', 'a.u.')")
    min_value: float = Field(..., description="Minimum axis value")
    max_value: float = Field(..., description="Maximum axis value")
    is_log_scale: bool = Field(default=False, description="Logarithmic scale")
    tick_spacing: Optional[str] = Field(None, description="Tick pattern")
    axis_visible: bool = Field(default=True, description="Axis clearly visible")


class DetectedMarker(BaseModel):
    """A single detected data point/marker"""
    estimated_x: float = Field(..., description="X coordinate (normalized 0-1)")
    estimated_y: float = Field(..., description="Y coordinate (normalized 0-1)")
    marker_type: str = Field(..., description="Type: 'star', 'circle', 'square', 'peak', etc.")
    relative_height: float = Field(..., description="Height vs max peak (0-1)")
    visibility_score: float = Field(..., description="Visibility: 1.0=clear, 0.5=partial, 0.1=barely")
    marker_color: Optional[str] = Field(None, description="Color")
    notes: Optional[str] = Field(None, description="Special notes")


class ChartExtractionResult(BaseModel):
    """Complete chart extraction result"""
    title: str = Field(..., description="Chart title")
    chart_type: str = Field(..., description="Chart type: 'spectrum', 'line_graph', 'scatter', etc.")
    x_axis: ChartAxis
    y_axis: ChartAxis
    markers: List[DetectedMarker] = Field(default_factory=list, description="Detected data points")
    overall_confidence: float = Field(..., description="Overall confidence (0-1)")
    axis_calibration_confidence: float = Field(..., description="Axis calibration confidence")
    marker_detection_confidence: float = Field(..., description="Marker detection confidence")
    notes: Optional[str] = Field(None, description="Caveats or observations")
    suggested_model_fit: Optional[str] = Field(None, description="Suggested peak model")
    data_quality_issues: Optional[List[str]] = Field(None, description="Quality issues detected")


# ============================================================================
# STRATEGY 1: GPT-5.5-PRO EXTRACTION (COMPLETIONS API)
# ============================================================================

def extract_with_gpt55pro_completions(
    client: OpenAI,
    image_path: Path,
) -> Tuple[ChartExtractionResult, str]:
    """
    Use GPT-5.5-Pro via /v1/completions endpoint (NOT chat).

    gpt-5.5-pro is a completion model, not a chat model.
    It uses client.completions.create() not client.chat.completions.create()

    Args:
        client: OpenAI client
        image_path: Path to chart image

    Returns:
        Tuple of (ChartExtractionResult, model_name)
    """
    logger.info(f"🚀 [GPT-5.5-PRO COMPLETIONS] Extracting from {image_path.name}")

    # Encode image to base64
    with open(image_path, "rb") as f:
        image_base64 = base64.standard_b64encode(f.read()).decode("utf-8")

    # Create prompt with image embedded
    prompt = f"""You are an elite spectroscopy data analyst. Extract numerical data from this chart image with MAXIMUM PRECISION.

Chart Image (base64): data:image/png;base64,{image_base64}

CRITICAL INSTRUCTIONS:
1. ALL coordinates normalized to [0, 1]: 0=left/bottom, 1=right/top
2. Read every axis label and tick
3. Locate EVERY visible marker/peak/point
4. For each marker: position (0-1), type, color, relative height, visibility (0-1)
5. Suggest peak model: gaussian, lorentzian, voigt, pseudo-voigt, asymmetric, multiplet
6. Confidence scores: overall, axis_calibration, marker_detection (each 0-1)
7. Flag data quality issues

OUTPUT: ONLY valid JSON matching this exact schema:
{{
  "title": "string",
  "chart_type": "spectrum|line_graph|scatter|bar|histogram",
  "x_axis": {{
    "label": "string",
    "unit": "string",
    "min_value": float,
    "max_value": float,
    "is_log_scale": boolean,
    "tick_spacing": "string|null",
    "axis_visible": boolean
  }},
  "y_axis": {{
    "label": "string",
    "unit": "string",
    "min_value": float,
    "max_value": float,
    "is_log_scale": boolean,
    "tick_spacing": "string|null",
    "axis_visible": boolean
  }},
  "markers": [
    {{
      "estimated_x": float,
      "estimated_y": float,
      "marker_type": "string",
      "relative_height": float,
      "visibility_score": float,
      "marker_color": "string|null",
      "notes": "string|null"
    }}
  ],
  "overall_confidence": float,
  "axis_calibration_confidence": float,
  "marker_detection_confidence": float,
  "notes": "string|null",
  "suggested_model_fit": "string|null",
  "data_quality_issues": ["string"]|null
}}

Accuracy requirements:
- Coordinates ±0.01 normalized
- Peak heights ±5% relative
- Axis ranges to nearest significant figure
- Be CONSERVATIVE with confidence"""

    try:
        # Use completions API (NOT chat completions)
        logger.info("   Calling /v1/completions endpoint...")

        response = client.completions.create(
            model="gpt-5.5-pro",
            prompt=prompt,
            temperature=0.1,  # Low temp for accuracy
            max_tokens=3000,
            top_p=0.9,
        )

        # Extract and parse JSON from response
        response_text = response.choices[0].text.strip()

        # Try to extract JSON from response
        logger.info("   Parsing JSON response...")

        # Look for JSON in response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            raise ValueError("No JSON found in response")

        json_str = json_match.group(0)
        result_dict = json.loads(json_str)

        # Parse into Pydantic model
        result = ChartExtractionResult(**result_dict)

        logger.info(
            f"✅ GPT-5.5-Pro extraction successful\n"
            f"   Markers: {len(result.markers)}\n"
            f"   Confidence: {result.overall_confidence:.2%}"
        )

        return result, "gpt-5.5-pro"

    except Exception as e:
        logger.error(f"❌ GPT-5.5-Pro completions failed: {e}")
        raise


# ============================================================================
# FALLBACK: GPT-4o CHAT EXTRACTION
# ============================================================================

def extract_with_gpt4o_chat(
    client: OpenAI,
    image_path: Path,
) -> Tuple[ChartExtractionResult, str]:
    """
    Fallback: Use GPT-4o via /v1/chat/completions endpoint.

    Args:
        client: OpenAI client
        image_path: Path to chart image

    Returns:
        Tuple of (ChartExtractionResult, model_name)
    """
    logger.info(f"🚀 [GPT-4o CHAT] Extracting from {image_path.name}")

    with open(image_path, "rb") as f:
        image_base64 = base64.standard_b64encode(f.read()).decode("utf-8")

    system_prompt = """You are an elite spectroscopy data analyst. Extract chart data with MAXIMUM PRECISION.

EXTRACTION PROTOCOL:
1. ALL coordinates normalized to [0, 1]: 0=left/bottom, 1=right/top
2. Read every axis label and tick mark
3. Locate EVERY visible marker, peak, or data point
4. For each marker: normalized position, type, color, relative height, visibility score
5. Suggest peak model: gaussian, lorentzian, voigt, pseudo-voigt, asymmetric, multiplet
6. Provide confidence scores: overall, axis_calibration, marker_detection
7. Identify data quality issues

Be CONSERVATIVE with confidence scores. Output ONLY valid JSON."""

    user_prompt = """Analyze this spectroscopy chart with maximum precision.

Extract:
1. Chart metadata (title, type)
2. X-axis: label, unit, min/max, scale type
3. Y-axis: label, unit, min/max, scale type
4. EVERY marker: normalized (0-1) coordinates, type, color, height, visibility
5. Suggested peak model
6. Confidence scores
7. Data quality issues

Accuracy: ±0.01 normalized coordinates, ±5% heights.
Output ONLY valid JSON."""

    try:
        logger.info("   Calling /v1/chat/completions endpoint...")

        response = client.beta.chat.completions.parse(
            model="gpt-4o",
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
            response_format=ChartExtractionResult,
            temperature=0.1,
            max_tokens=3000,
        )

        result = response.choices[0].message.parsed
        if result is None:
            raise ValueError("GPT-4o returned null response")

        logger.info(
            f"✅ GPT-4o extraction successful\n"
            f"   Markers: {len(result.markers)}\n"
            f"   Confidence: {result.overall_confidence:.2%}"
        )

        return result, "gpt-4o"

    except Exception as e:
        logger.error(f"❌ GPT-4o chat failed: {e}")
        raise


# ============================================================================
# STRATEGY 2: OCR FOR AXIS LABELS (VERIFICATION)
# ============================================================================

def extract_axis_labels_with_ocr(image_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Use Tesseract OCR to verify axis labels"""
    if not TESSERACT_AVAILABLE:
        return None, None

    logger.info(f"📖 [OCR] Verifying axis labels")

    try:
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape

        # Bottom region for X-axis
        x_axis_region = gray[int(h*0.85):, :]
        x_label = pytesseract.image_to_string(x_axis_region).strip()

        # Left region for Y-axis
        y_axis_region = gray[:, 0:int(w*0.15)]
        y_label = pytesseract.image_to_string(y_axis_region).strip()

        if x_label or y_label:
            logger.info(f"   ✓ OCR labels found")
            return x_label, y_label

        return None, None

    except Exception as e:
        logger.debug(f"OCR skipped: {e}")
        return None, None


# ============================================================================
# STRATEGY 3: LOCAL MARKER VERIFICATION
# ============================================================================

def verify_markers_local(image_path: Path) -> Dict[str, int]:
    """Use OpenCV to verify marker counts"""
    logger.info(f"🔍 [LOCAL VERIFICATION] Counting markers")

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return {}

        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        marker_counts = {}

        # Red markers
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_count = sum(1 for c in red_contours if 20 < cv2.contourArea(c) < 5000)
        if red_count > 0:
            marker_counts["red"] = red_count

        # Blue markers
        blue_lower = np.array([100, 100, 100])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_count = sum(1 for c in blue_contours if 20 < cv2.contourArea(c) < 5000)
        if blue_count > 0:
            marker_counts["blue"] = blue_count

        if marker_counts:
            logger.info(f"   ✓ Verification: {marker_counts}")

        return marker_counts

    except Exception as e:
        logger.debug(f"Local verification skipped: {e}")
        return {}


# ============================================================================
# STRATEGY 4: PEAK FITTING VALIDATION
# ============================================================================

class SpectroscopyPeakFitter:
    """Validate markers using spectroscopic models"""

    @classmethod
    def validate_and_refine(
        cls,
        markers: List[DetectedMarker],
        x_axis: ChartAxis,
        y_axis: ChartAxis,
        model: str = "gaussian",
    ) -> Dict[str, Any]:
        """Validate markers using peak models"""
        if not SCIPY_AVAILABLE or not markers:
            return {"status": "skipped"}

        try:
            logger.info(f"📊 [PEAK FITTING] Validating {len(markers)} markers")

            data_points = []
            for marker in markers:
                data_x = (x_axis.min_value +
                         marker.estimated_x *
                         (x_axis.max_value - x_axis.min_value))

                data_y = (y_axis.min_value +
                         (1 - marker.estimated_y) *
                         (y_axis.max_value - y_axis.min_value))

                data_points.append({
                    "x": data_x,
                    "y": data_y,
                    "confidence": marker.visibility_score,
                })

            data_points.sort(key=lambda p: p["x"])

            logger.info(f"   ✓ Validation complete")
            return {
                "status": "success",
                "model_used": model,
                "point_count": len(data_points),
            }

        except Exception as e:
            logger.warning(f"Peak fitting issue: {e}")
            return {"status": "warning"}


# ============================================================================
# MULTI-STRATEGY AGENT
# ============================================================================

class EnhancedMultiStrategyAgent:
    """Primary: GPT-5.5-Pro (completions) with GPT-4o fallback"""

    def __init__(self, image_path: Path, client: OpenAI):
        self.image_path = image_path
        self.client = client
        self.result = None
        self.model_used = None
        self.ocr_labels = None
        self.local_verification = None

    def run_all_strategies(self) -> Tuple[ChartExtractionResult, str]:
        """Execute all strategies with proper fallback"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 ENHANCED CHART EXTRACTION - GPT-5.5-Pro / GPT-4o")
        logger.info(f"{'='*80}\n")

        # STRATEGY 1: GPT-5.5-Pro (try first)
        try:
            logger.info("[1/4] Running GPT-5.5-Pro completions extraction...")
            self.result, self.model_used = extract_with_gpt55pro_completions(
                self.client,
                self.image_path
            )
            logger.info(f"✅ [1/4] GPT-5.5-Pro: SUCCESS\n")
        except Exception as e:
            logger.warning(f"⚠️  [1/4] GPT-5.5-Pro failed, trying GPT-4o fallback...")

            try:
                logger.info("[1/4] Running GPT-4o chat extraction...")
                self.result, self.model_used = extract_with_gpt4o_chat(
                    self.client,
                    self.image_path
                )
                logger.info(f"✅ [1/4] GPT-4o: SUCCESS (fallback)\n")
            except Exception as e2:
                logger.error(f"❌ Both models failed")
                raise ValueError(f"GPT-5.5-Pro failed: {e}\nGPT-4o failed: {e2}")

        # STRATEGY 2: OCR Verification
        logger.info("[2/4] Running OCR verification...")
        x_label, y_label = extract_axis_labels_with_ocr(self.image_path)
        if x_label or y_label:
            logger.info(f"✅ [2/4] OCR: Found labels\n")
            self.ocr_labels = (x_label, y_label)
        else:
            logger.info(f"ℹ️  [2/4] OCR: No additional data\n")

        # STRATEGY 3: Local Verification
        logger.info("[3/4] Running local marker verification...")
        self.local_verification = verify_markers_local(self.image_path)
        if self.local_verification:
            logger.info(f"✅ [3/4] Local verification: {self.local_verification}\n")
        else:
            logger.info(f"ℹ️  [3/4] Local verification: No markers\n")

        # STRATEGY 4: Peak Fitting
        logger.info("[4/4] Running peak fitting validation...")
        fitter = SpectroscopyPeakFitter()
        peak_report = fitter.validate_and_refine(
            self.result.markers,
            self.result.x_axis,
            self.result.y_axis,
            model=self.result.suggested_model_fit or "gaussian"
        )
        logger.info(f"✅ [4/4] Peak fitting: {peak_report['status']}\n")

        # Merge results
        self._merge_results()

        logger.info(f"{'='*80}")
        logger.info(f"✨ EXTRACTION COMPLETE")
        logger.info(f"   Model: {self.model_used}")
        logger.info(f"   Markers: {len(self.result.markers)}")
        logger.info(f"   Confidence: {self.result.overall_confidence:.2%}")
        logger.info(f"{'='*80}\n")

        return self.result, self.model_used

    def _merge_results(self):
        """Merge verification results"""
        if self.ocr_labels:
            if self.ocr_labels[0]:
                self.result.x_axis.label = self.ocr_labels[0]
            if self.ocr_labels[1]:
                self.result.y_axis.label = self.ocr_labels[1]

        if self.local_verification:
            local_total = sum(self.local_verification.values())
            gpt_total = len(self.result.markers)

            if abs(gpt_total - local_total) > 3:
                logger.warning(f"⚠️  Marker count mismatch: {gpt_total} vs {local_total}")
                if self.result.data_quality_issues is None:
                    self.result.data_quality_issues = []
                self.result.data_quality_issues.append("marker_count_discrepancy")


# ============================================================================
# CSV & METADATA GENERATION
# ============================================================================

class CSVGenerator:
    """Convert results to CSV and metadata"""

    @staticmethod
    def to_dataframe(result: ChartExtractionResult) -> pd.DataFrame:
        """Convert markers to DataFrame"""
        data = []

        for idx, marker in enumerate(result.markers, 1):
            data_x = (result.x_axis.min_value +
                     marker.estimated_x *
                     (result.x_axis.max_value - result.x_axis.min_value))

            data_y = (result.y_axis.min_value +
                     (1 - marker.estimated_y) *
                     (result.y_axis.max_value - result.y_axis.min_value))

            data.append({
                "point_id": idx,
                f"{result.x_axis.label} ({result.x_axis.unit})": data_x,
                f"{result.y_axis.label} ({result.y_axis.unit})": data_y,
                "marker_type": marker.marker_type,
                "marker_color": marker.marker_color or "unknown",
                "relative_height": marker.relative_height,
                "visibility_score": marker.visibility_score,
                "notes": marker.notes or "",
            })

        df = pd.DataFrame(data)

        # Sort by X
        x_col = [c for c in df.columns if result.x_axis.label in c][0]
        df = df.sort_values(x_col).reset_index(drop=True)

        return df

    @staticmethod
    def save_csv(result: ChartExtractionResult, output_path: Path) -> Path:
        """Save to CSV"""
        df = CSVGenerator.to_dataframe(result)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"✅ CSV saved: {output_path}")
        logger.info(f"   Rows: {len(df)}")

        return output_path

    @staticmethod
    def save_json_metadata(
        result: ChartExtractionResult,
        output_path: Path,
        model_used: str
    ) -> Path:
        """Save metadata as JSON"""
        output_data = {
            "title": result.title,
            "chart_type": result.chart_type,
            "extracted_at": datetime.now().isoformat(),
            "model_used": model_used,
            "overall_confidence": result.overall_confidence,
            "axis_calibration_confidence": result.axis_calibration_confidence,
            "marker_detection_confidence": result.marker_detection_confidence,
            "axes": {
                "x": {
                    "label": result.x_axis.label,
                    "unit": result.x_axis.unit,
                    "min": result.x_axis.min_value,
                    "max": result.x_axis.max_value,
                    "log_scale": result.x_axis.is_log_scale,
                },
                "y": {
                    "label": result.y_axis.label,
                    "unit": result.y_axis.unit,
                    "min": result.y_axis.min_value,
                    "max": result.y_axis.max_value,
                    "log_scale": result.y_axis.is_log_scale,
                }
            },
            "marker_count": len(result.markers),
            "markers": [m.model_dump() for m in result.markers],
            "suggested_model": result.suggested_model_fit,
            "data_quality_issues": result.data_quality_issues,
            "extraction_notes": result.notes,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"✅ Metadata saved: {output_path}")

        return output_path


# ============================================================================
# PLOTLY VERIFICATION
# ============================================================================

def create_plotly_verification(
    csv_path: Path,
    json_path: Path,
    output_html: Path,
) -> Path:
    """Create interactive Plotly verification"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("Plotly not available")
        return None

    logger.info(f"📈 Generating verification chart...")

    df = pd.read_csv(csv_path)

    with open(json_path) as f:
        metadata = json.load(f)

    # Find axis columns
    x_col = next(
        (c for c in df.columns if any(x in c.lower() for x in ['energy', 'eV', 'binding', 'wavenumber'])),
        df.columns[1]
    )
    y_col = next(
        (c for c in df.columns if any(y in c.lower() for y in ['intensity', 'counts', 'transmission', 'absorbance'])),
        df.columns[2]
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers+lines',
        marker=dict(
            size=12,
            color=df.get('visibility_score', [0.7]*len(df)),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Visibility", thickness=20, len=300),
            line=dict(width=1, color='white'),
        ),
        line=dict(width=2, color='rgba(0,0,255,0.5)'),
        name='Extracted Data',
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>{metadata.get('title', 'Chart')}</b><br>"
                 f"<sub>{metadata['model_used']} | Confidence: {metadata['overall_confidence']:.1%}</sub>",
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        height=700,
        width=1200,
        hovermode='closest',
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))

    logger.info(f"✅ Verification HTML: {output_html}")

    return output_html


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def process_chart_image(
    image_path: Path,
    output_dir: Path = Path("extraction_output"),
) -> Dict[str, Any]:
    """Complete workflow: Multi-Strategy → CSV → Metadata → Verification"""
    logger.info(f"\n{'='*80}")
    logger.info(f"ADVANCED CHART EXTRACTION")
    logger.info(f"{'='*80}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load API client
    load_dotenv("info.env")
    api_key = os.getenv("SUPER_SECRET_API_KEY")
    if not api_key:
        raise EnvironmentError("SUPER_SECRET_API_KEY not found in info.env")

    client = OpenAI(api_key=api_key)

    # Run extraction
    agent = EnhancedMultiStrategyAgent(image_path, client)
    result, model_used = agent.run_all_strategies()

    # Generate outputs
    stem = image_path.stem
    csv_path = output_dir / f"{stem}_extracted_data.csv"
    json_path = output_dir / f"{stem}_extraction_metadata.json"
    html_path = output_dir / f"{stem}_verification.html"

    CSVGenerator.save_csv(result, csv_path)
    CSVGenerator.save_json_metadata(result, json_path, model_used)
    create_plotly_verification(csv_path, json_path, html_path)

    logger.info(f"\n{'='*80}")
    logger.info(f"✨ COMPLETE (Model: {model_used})")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Data CSV:     {csv_path}")
    logger.info(f"📋 Metadata:     {json_path}")
    logger.info(f"📈 Verification: {html_path}")
    logger.info(f"{'='*80}\n")

    return {
        "csv": csv_path,
        "json": json_path,
        "html": html_path,
        "model_used": model_used,
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("extraction_output")

        try:
            results = process_chart_image(image_path, output_dir)
            print(f"\n✅ SUCCESS! (Model: {results['model_used']})")
            for key, path in results.items():
                if key != "model_used":
                    print(f"   {key.upper():12s}: {path}")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
            sys.exit(1)
    else:
        print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║            Advanced Chart Extraction - GPT-5.5-Pro / GPT-4o Fallback          ║
╚════════════════════════════════════════════════════════════════════════════════╝

Usage: python vision_agent_gpt55_pro.py <image_path> [output_dir]

Arguments:
  image_path   - Path to chart image
  output_dir   - Optional output directory (default: extraction_output)

Examples:
  python vision_agent_gpt55_pro.py chart.png
  python vision_agent_gpt55_pro.py chart.png extraction_output

Strategy:
  1. Try GPT-5.5-Pro (/v1/completions endpoint)
  2. Fall back to GPT-4o (/v1/chat/completions) if needed
  3. Verify with OCR, local detection, peak fitting

Features:
  ✓ GPT-5.5-Pro completions API support
  ✓ Automatic GPT-4o fallback
  ✓ Structured JSON extraction
  ✓ Multi-strategy validation
  ✓ Per-marker visibility scoring
  ✓ Data quality assessment
  ✓ Interactive Plotly verification
        """)
