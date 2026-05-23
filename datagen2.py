"""
Training Data Generator with Peaks for Photoelectron Spectroscopy
Generates synthetic training data with Gaussian peaks, outputs CSV with all points,
and identifies peak ranges with detailed annotations in units.

Generates two separate visualizations:
1. Clean graph visualization (graph only)
2. Detailed analysis (with peak coordinates and statistics table)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path
import argparse


@dataclass
class PeakInfo:
    """Stores information about a single peak"""
    peak_id: int
    center_x: float
    center_y: float
    range_start: float
    range_end: float
    width: float
    sigma: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DataStatistics:
    """Stores statistics about generated data"""
    min_y: float
    max_y: float
    mean_y: float
    std_y: float
    min_x: float
    max_x: float

    def to_dict(self) -> Dict:
        return asdict(self)


class PhotoelectronSpectroscopyDataGenerator:
    """Generate synthetic photoelectron spectroscopy training data with peaks"""

    def __init__(
            self,
            num_points: int = 500,
            num_peaks: int = 5,
            peak_height: float = 100.0,
            noise_level: float = 0.1,
            x_min: float = 0.0,
            x_max: float = 100.0,
            seed: int = None,
            peak_placement: str = 'random',  # 'random' or 'uniform'
    ):
        """
        Initialize the photoelectron spectroscopy data generator.

        Args:
            num_points: Number of data points to generate
            num_peaks: Number of peaks to generate
            peak_height: Maximum peak height (units)
            noise_level: Noise level as fraction of peak height (0.0-1.0)
            x_min: Minimum x coordinate (binding energy, eV)
            x_max: Maximum x coordinate (binding energy, eV)
            seed: Random seed for reproducibility
            peak_placement: 'random' for random placement or 'uniform' for evenly spaced peaks
        """
        self.num_points = num_points
        self.num_peaks = num_peaks
        self.peak_height = peak_height
        self.noise_level = noise_level
        self.x_min = x_min
        self.x_max = x_max
        self.peak_placement = peak_placement

        if seed is not None:
            np.random.seed(seed)

        self.x = None
        self.y = None
        self.y_before_noise = None  # Store data before noise for accurate peak detection
        self.peaks: List[PeakInfo] = []
        self.statistics: DataStatistics = None
        self.df: pd.DataFrame = None

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the photoelectron spectroscopy training data with peaks.

        Returns:
            Tuple of (x_coordinates, y_values)
        """
        # Generate x values (binding energy)
        self.x = np.linspace(self.x_min, self.x_max, self.num_points)

        # Initialize y values (intensity/counts)
        self.y = np.zeros_like(self.x)

        # Calculate sigma for gaussian peaks (width of peaks)
        sigma = (self.x_max - self.x_min) / (self.num_peaks * 5)

        # Generate peak positions
        if self.peak_placement == 'uniform':
            # Evenly spaced peaks for more predictable, accurate placement
            peak_x_values = np.linspace(
                self.x_min + (self.x_max - self.x_min) * 0.1,
                self.x_max - (self.x_max - self.x_min) * 0.1,
                self.num_peaks
            )
            # Add small random jitter to avoid perfect uniformity
            jitter = np.random.uniform(-sigma * 0.3, sigma * 0.3, self.num_peaks)
            peak_x_values = np.clip(peak_x_values + jitter, self.x_min, self.x_max)
        else:
            # Random placement
            peak_x_values = np.random.uniform(self.x_min, self.x_max, self.num_peaks)
            peak_x_values = np.sort(peak_x_values)

        # Generate peaks and store original peak centers
        peak_centers_data = []
        for peak_id, peak_x in enumerate(peak_x_values, start=1):
            # Create Gaussian peak
            gaussian = self.peak_height * np.exp(
                -((self.x - peak_x) ** 2) / (2 * sigma ** 2)
            )
            self.y += gaussian
            peak_centers_data.append({
                'peak_id': peak_id,
                'original_x': peak_x,
                'sigma': sigma
            })

        # Store data before noise for accurate peak detection
        self.y_before_noise = self.y.copy()

        # Add Gaussian noise
        noise = np.random.normal(
            0,
            self.noise_level * self.peak_height,
            self.num_points
        )
        self.y = self.y + noise

        # Ensure no negative values
        self.y = np.maximum(self.y, 0)

        # Detect actual peak centers for improved accuracy
        self._detect_and_store_peaks(peak_centers_data, sigma)

        # Calculate statistics
        self._calculate_statistics()

        # Create DataFrame
        self._create_dataframe()

        return self.x, self.y

    def _detect_and_store_peaks(self, peak_centers_data: List[Dict], sigma: float) -> None:
        """
        Detect actual peak centers using signal processing on clean data for improved accuracy.

        Uses the data BEFORE noise was added to detect true peaks, avoiding noise artifacts.
        Then retrieves the corresponding values from the noisy data.

        Args:
            peak_centers_data: List of dictionaries with original peak information
            sigma: Standard deviation of the peaks
        """
        # Use the clean data (before noise) to detect peaks accurately
        # This prevents noise-induced local maxima from being selected
        min_height = self.peak_height * 0.25  # Peaks must be at least 25% of max height
        min_distance = int(sigma / ((self.x_max - self.x_min) / self.num_points))

        # Detect peaks in clean data
        detected_indices, properties = find_peaks(
            self.y_before_noise,
            height=min_height,
            distance=min_distance
        )

        # If we found enough peaks, use the detected ones
        if len(detected_indices) >= self.num_peaks:
            # Get the heights of detected peaks
            peak_heights = properties['peak_heights']
            
            # Select the top num_peaks by height (these are the major peaks)
            top_indices = np.argsort(peak_heights)[-self.num_peaks:]
            top_indices = detected_indices[top_indices]
            top_indices = np.sort(top_indices)

            for peak_id, peak_idx in enumerate(top_indices, start=1):
                peak_x = self.x[peak_idx]
                # Use the NOISY data for the peak value to be realistic
                peak_y = self.y[peak_idx]

                # Calculate peak range (3-sigma rule)
                range_start = max(self.x_min, peak_x - 3 * sigma)
                range_end = min(self.x_max, peak_x + 3 * sigma)

                self.peaks.append(PeakInfo(
                    peak_id=peak_id,
                    center_x=float(peak_x),
                    center_y=float(peak_y),
                    range_start=float(range_start),
                    range_end=float(range_end),
                    width=float(range_end - range_start),
                    sigma=float(sigma),
                ))
        else:
            # Fallback to original positions if peak detection finds too few peaks
            # This handles edge cases with very high noise or unusual distributions
            for idx, data in enumerate(peak_centers_data):
                peak_id = data['peak_id']
                peak_x = data['original_x']
                peak_idx = np.argmin(np.abs(self.x - peak_x))
                peak_y = self.y[peak_idx]

                range_start = max(self.x_min, peak_x - 3 * sigma)
                range_end = min(self.x_max, peak_x + 3 * sigma)

                self.peaks.append(PeakInfo(
                    peak_id=peak_id,
                    center_x=float(peak_x),
                    center_y=float(peak_y),
                    range_start=float(range_start),
                    range_end=float(range_end),
                    width=float(range_end - range_start),
                    sigma=float(sigma),
                ))

    def _calculate_statistics(self) -> None:
        """Calculate and store data statistics"""
        self.statistics = DataStatistics(
            min_y=float(np.min(self.y)),
            max_y=float(np.max(self.y)),
            mean_y=float(np.mean(self.y)),
            std_y=float(np.std(self.y)),
            min_x=float(np.min(self.x)),
            max_x=float(np.max(self.x)),
        )

    def _create_dataframe(self) -> None:
        """Create pandas DataFrame with all data points"""
        self.df = pd.DataFrame({
            'point_id': np.arange(1, self.num_points + 1),
            'binding_energy_ev': self.x,
            'intensity_counts': self.y,
        })

    def save_csv(self, filepath: str) -> str:
        """
        Save generated data to CSV file.

        Args:
            filepath: Path where CSV should be saved

        Returns:
            Path to saved CSV file
        """
        if self.df is None:
            raise ValueError("No data generated. Call generate() first.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(filepath, index=False)
        return filepath

    def save_peak_ranges(self, filepath: str) -> str:
        """
        Save peak information to JSON file.

        Args:
            filepath: Path where JSON should be saved

        Returns:
            Path to saved JSON file
        """
        if not self.peaks:
            raise ValueError("No peaks generated. Call generate() first.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'generated_at': datetime.now().isoformat(),
            'analysis_type': 'Photoelectron Spectroscopy',
            'configuration': {
                'num_points': self.num_points,
                'num_peaks': self.num_peaks,
                'peak_height': self.peak_height,
                'noise_level': self.noise_level,
                'binding_energy_range_ev': [self.x_min, self.x_max],
                'peak_placement': self.peak_placement,
            },
            'peaks': [peak.to_dict() for peak in self.peaks],
            'statistics': self.statistics.to_dict(),
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)

        return filepath

    def visualize_graph_only(self, filepath: str = None) -> None:
        """
        Create a clean graph visualization (graph only, no tables).

        Args:
            filepath: Optional path to save the figure
        """
        if self.y is None:
            raise ValueError("No data generated. Call generate() first.")

        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot the main data
        ax.plot(self.x, self.y, 'b-', linewidth=2.5, label='Photoelectron Spectrum', alpha=0.85)

        # Plot peak centers
        peak_centers_x = [peak.center_x for peak in self.peaks]
        peak_centers_y = [peak.center_y for peak in self.peaks]
        ax.scatter(peak_centers_x, peak_centers_y, color='red', s=200,
                   zorder=5, marker='*', label='Peak Centers', edgecolors='darkred', linewidth=2)

        # Highlight peak ranges with subtle shading
        for i, peak in enumerate(self.peaks):
            ax.axvspan(peak.range_start, peak.range_end, alpha=0.12,
                       color='red', zorder=1)

            # Add peak labels at the top of each peak
            ax.text(peak.center_x, peak.center_y + (self.statistics.max_y * 0.05),
                    f"P{peak.peak_id}",
                    ha='center', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow',
                              alpha=0.8, edgecolor='black', linewidth=1.5))

        ax.set_xlabel('Binding Energy (eV)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Intensity (Counts)', fontsize=13, fontweight='bold')
        ax.set_title('Photoelectron Spectroscopy - Synthetic Training Data',
                     fontsize=15, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.95)

        # Improve layout
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=11)

        plt.tight_layout()

        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Clean graph visualization saved to {filepath}")

        plt.show()

    def visualize_detailed_analysis(self, filepath: str = None) -> None:
        """
        Create detailed visualization with peak coordinates and statistics table.

        Args:
            filepath: Optional path to save the figure
        """
        if self.y is None:
            raise ValueError("No data generated. Call generate() first.")

        fig = plt.figure(figsize=(16, 12))

        # Main plot with data and peaks
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(self.x, self.y, 'b-', linewidth=2.5, label='Photoelectron Spectrum', alpha=0.85)

        # Plot peak centers
        peak_centers_x = [peak.center_x for peak in self.peaks]
        peak_centers_y = [peak.center_y for peak in self.peaks]
        ax1.scatter(peak_centers_x, peak_centers_y, color='red', s=200,
                    zorder=5, marker='*', label='Peak Centers', edgecolors='darkred', linewidth=2)

        # Highlight peak ranges
        for i, peak in enumerate(self.peaks):
            ax1.axvspan(peak.range_start, peak.range_end, alpha=0.12, color='red')
            ax1.text(peak.center_x, peak.center_y + 10, f"P{peak.peak_id}",
                     ha='center', fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        ax1.set_xlabel('Binding Energy (eV)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Intensity (Counts)', fontsize=12, fontweight='bold')
        ax1.set_title('Photoelectron Spectroscopy - Full Analysis',
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', fontsize=11)

        # Peak ranges detail plot
        ax2 = plt.subplot(3, 1, 2)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.peaks)))

        for peak, color in zip(self.peaks, colors):
            # Draw range as a bar
            ax2.barh(peak.peak_id - 0.4, peak.width, left=peak.range_start,
                     height=0.8, color=color, alpha=0.8, edgecolor='black', linewidth=2)
            # Add center marker
            ax2.plot(peak.center_x, peak.peak_id, 'r*', markersize=20, zorder=5)
            # Add text label
            ax2.text(peak.range_start - (self.x_max - self.x_min) * 0.03,
                     peak.peak_id, f"P{peak.peak_id}",
                     va='center', ha='right', fontweight='bold', fontsize=11)

        ax2.set_xlabel('Binding Energy (eV)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Peak ID', fontsize=12, fontweight='bold')
        ax2.set_title('Peak Range Visualization', fontsize=14, fontweight='bold')
        ax2.set_yticks(range(1, len(self.peaks) + 1))
        ax2.grid(True, alpha=0.3, axis='x', linestyle='--')

        # Peak information table
        ax3 = plt.subplot(3, 1, 3)
        ax3.axis('off')

        table_data = [[
            'Peak ID', 'Center Energy\n(eV)', 'Intensity\n(Counts)',
            'Range Start\n(eV)', 'Range End\n(eV)', 'Width\n(eV)', 'Sigma\n(eV)'
        ]]
        for peak in self.peaks:
            table_data.append([
                f"{peak.peak_id}",
                f"{peak.center_x:.4f}",
                f"{peak.center_y:.2f}",
                f"{peak.range_start:.4f}",
                f"{peak.range_end:.4f}",
                f"{peak.width:.4f}",
                f"{peak.sigma:.4f}",
            ])

        table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.09, 0.13, 0.13, 0.14, 0.14, 0.13, 0.12])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.8)

        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#e3f2fd')
                else:
                    table[(i, j)].set_facecolor('#ffffff')

        ax3.text(0.5, 0.98, 'Detailed Peak Coordinate Information',
                 transform=ax3.transAxes, ha='center', fontsize=13,
                 fontweight='bold', va='top')

        # Add statistics box
        stats_text = (
            f"Statistics:\n"
            f"Min Intensity: {self.statistics.min_y:.2f} | "
            f"Max Intensity: {self.statistics.max_y:.2f}\n"
            f"Mean Intensity: {self.statistics.mean_y:.2f} | "
            f"Std Dev: {self.statistics.std_y:.2f}"
        )
        ax3.text(0.5, -0.15, stats_text,
                 transform=ax3.transAxes, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()

        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Detailed analysis visualization saved to {filepath}")

        plt.show()

    def print_summary(self) -> None:
        """Print a summary of the generated data and peaks"""
        print("\n" + "=" * 75)
        print("PHOTOELECTRON SPECTROSCOPY DATA GENERATION SUMMARY".center(75))
        print("=" * 75)

        print(f"\n📊 Configuration:")
        print(f"  • Total Data Points: {self.num_points}")
        print(f"  • Number of Peaks: {self.num_peaks}")
        print(f"  • Maximum Peak Intensity: {self.peak_height} counts")
        print(f"  • Noise Level: {self.noise_level}")
        print(f"  • Peak Placement: {self.peak_placement}")
        print(f"  • Binding Energy Range: {self.x_min:.2f} - {self.x_max:.2f} eV")

        print(f"\n📈 Data Statistics:")
        print(f"  • Intensity Range: {self.statistics.min_y:.4f} - {self.statistics.max_y:.4f} counts")
        print(f"  • Mean Intensity: {self.statistics.mean_y:.4f} counts")
        print(f"  • Standard Deviation: {self.statistics.std_y:.4f} counts")

        print(f"\n🎯 Peak Information:")
        for peak in self.peaks:
            print(f"  Peak {peak.peak_id}:")
            print(f"    - Center: ({peak.center_x:.4f} eV, {peak.center_y:.2f} counts)")
            print(f"    - Binding Energy Range: [{peak.range_start:.4f}, {peak.range_end:.4f}] eV")
            print(f"    - Peak Width: {peak.width:.4f} eV")
            print(f"    - Standard Deviation (σ): {peak.sigma:.4f} eV")

        print("\n" + "=" * 75 + "\n")


class BatchDatasetGenerator:
    """Generate multiple datasets in batch with varying parameters"""

    def __init__(self, base_output_dir: str = 'training_data_batch/'):
        """
        Initialize batch generator.

        Args:
            base_output_dir: Base directory for all batch outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_metadata = []

    def generate_batch(
            self,
            batch_size: int = 10,
            num_points: int = 500,
            num_peaks: int = 5,
            peak_height_range: Tuple[float, float] = (80.0, 120.0),
            noise_level_range: Tuple[float, float] = (0.05, 0.15),
            x_min: float = 0.0,
            x_max: float = 100.0,
            peak_placement: str = 'uniform',
            save_visualizations: bool = False,
            verbose: bool = True,
    ) -> str:
        """
        Generate a batch of datasets with varying parameters.

        Args:
            batch_size: Number of datasets to generate
            num_points: Number of data points per dataset
            num_peaks: Number of peaks per dataset
            peak_height_range: Range of peak heights (min, max)
            noise_level_range: Range of noise levels (min, max)
            x_min: Minimum binding energy
            x_max: Maximum binding energy
            peak_placement: 'random' or 'uniform'
            save_visualizations: Whether to save visualization for each dataset
            verbose: Whether to print progress

        Returns:
            Path to batch directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_dir = self.base_output_dir / f'batch_{timestamp}'
        batch_dir.mkdir(parents=True, exist_ok=True)

        batch_info = {
            'timestamp': timestamp,
            'batch_size': batch_size,
            'parameters': {
                'num_points': num_points,
                'num_peaks': num_peaks,
                'peak_height_range': peak_height_range,
                'noise_level_range': noise_level_range,
                'binding_energy_range': [x_min, x_max],
                'peak_placement': peak_placement,
            },
            'datasets': []
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Generating batch of {batch_size} datasets")
            print(f"Output directory: {batch_dir}")
            print(f"{'=' * 60}\n")

        for i in range(batch_size):
            # Vary parameters within specified ranges
            peak_height = np.random.uniform(peak_height_range[0], peak_height_range[1])
            noise_level = np.random.uniform(noise_level_range[0], noise_level_range[1])

            # Generate dataset
            generator = PhotoelectronSpectroscopyDataGenerator(
                num_points=num_points,
                num_peaks=num_peaks,
                peak_height=peak_height,
                noise_level=noise_level,
                x_min=x_min,
                x_max=x_max,
                seed=None,  # Use random seed for variation
                peak_placement=peak_placement,
            )

            generator.generate()

            # Create subdirectory for this dataset
            dataset_dir = batch_dir / f'dataset_{i+1:03d}'
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Save files
            csv_path = dataset_dir / 'training_data.csv'
            json_path = dataset_dir / 'peak_ranges.json'

            generator.save_csv(str(csv_path))
            generator.save_peak_ranges(str(json_path))

            # Save visualizations if requested
            if save_visualizations:
                graph_path = dataset_dir / 'graph_only.png'
                generator.visualize_graph_only(str(graph_path))

            # Record metadata
            dataset_metadata = {
                'dataset_id': i + 1,
                'directory': str(dataset_dir.relative_to(self.base_output_dir)),
                'parameters': {
                    'peak_height': peak_height,
                    'noise_level': noise_level,
                },
                'files': {
                    'csv': 'training_data.csv',
                    'json': 'peak_ranges.json',
                    'graph': 'graph_only.png' if save_visualizations else None,
                },
                'num_peaks_detected': len(generator.peaks),
            }
            batch_info['datasets'].append(dataset_metadata)

            if verbose:
                print(f"✓ Dataset {i+1:3d}/{batch_size} generated "
                      f"(h={peak_height:.1f}, σ_n={noise_level:.3f})")

        # Save batch metadata
        metadata_path = batch_dir / 'batch_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(batch_info, f, indent=2)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"✓ Batch generation complete!")
            print(f"✓ Metadata saved: {metadata_path}")
            print(f"{'=' * 60}\n")

        return str(batch_dir)

    def generate_batch_with_variations(
            self,
            batch_size: int = 10,
            num_points: int = 500,
            num_peaks_list: Optional[List[int]] = None,
            peak_height: float = 100.0,
            noise_level: float = 0.1,
            x_min: float = 0.0,
            x_max: float = 100.0,
            save_visualizations: bool = False,
            verbose: bool = True,
    ) -> str:
        """
        Generate batch with specific variations (e.g., different number of peaks).

        Args:
            batch_size: Datasets per variation
            num_points: Number of data points
            num_peaks_list: List of peak counts to vary
            peak_height: Peak height
            noise_level: Noise level
            x_min: Minimum binding energy
            x_max: Maximum binding energy
            save_visualizations: Whether to save visualizations
            verbose: Whether to print progress

        Returns:
            Path to batch directory
        """
        if num_peaks_list is None:
            num_peaks_list = [3, 5, 7]

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_dir = self.base_output_dir / f'batch_variations_{timestamp}'
        batch_dir.mkdir(parents=True, exist_ok=True)

        total_datasets = batch_size * len(num_peaks_list)
        dataset_count = 0

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Generating {total_datasets} datasets with peak variations")
            print(f"Peak counts: {num_peaks_list}")
            print(f"Datasets per variation: {batch_size}")
            print(f"Output directory: {batch_dir}")
            print(f"{'=' * 60}\n")

        for num_peaks in num_peaks_list:
            variation_dir = batch_dir / f'peaks_{num_peaks}'
            variation_dir.mkdir(parents=True, exist_ok=True)

            if verbose:
                print(f"\nGenerating {batch_size} datasets with {num_peaks} peaks:")

            for i in range(batch_size):
                generator = PhotoelectronSpectroscopyDataGenerator(
                    num_points=num_points,
                    num_peaks=num_peaks,
                    peak_height=peak_height,
                    noise_level=noise_level,
                    x_min=x_min,
                    x_max=x_max,
                    seed=None,
                    peak_placement='uniform',
                )

                generator.generate()

                dataset_dir = variation_dir / f'dataset_{i+1:03d}'
                dataset_dir.mkdir(parents=True, exist_ok=True)

                csv_path = dataset_dir / 'training_data.csv'
                json_path = dataset_dir / 'peak_ranges.json'

                generator.save_csv(str(csv_path))
                generator.save_peak_ranges(str(json_path))

                if save_visualizations:
                    graph_path = dataset_dir / 'graph_only.png'
                    generator.visualize_graph_only(str(graph_path))

                dataset_count += 1
                if verbose:
                    print(f"  ✓ Dataset {i+1:3d}/{batch_size}")

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"✓ Generated {total_datasets} datasets total!")
            print(f"{'=' * 60}\n")

        return str(batch_dir)


def main():
    """Command-line interface for the photoelectron spectroscopy data generator"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic photoelectron spectroscopy training data with Gaussian peaks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default settings
  python datagen2.py

  # Generate 1000 points with 8 peaks using uniform placement
  python datagen2.py --num-points 1000 --num-peaks 8 --peak-placement uniform

  # Generate with custom parameters
  python datagen2.py --num-points 500 --peak-height 150 --output data/

  # Generate batch of 20 datasets
  python datagen2.py --batch --batch-size 20

  # Generate batch with peak variations
  python datagen2.py --batch-variations --batch-size 5 --num-peaks-list 3 5 7
        """
    )

    # Single dataset arguments
    parser.add_argument('--num-points', type=int, default=500,
                        help='Number of data points (default: 500)')
    parser.add_argument('--num-peaks', type=int, default=5,
                        help='Number of peaks (default: 5)')
    parser.add_argument('--peak-height', type=float, default=100.0,
                        help='Maximum peak intensity in counts (default: 100)')
    parser.add_argument('--noise-level', type=float, default=0.1,
                        help='Noise level 0.0-1.0 (default: 0.1)')
    parser.add_argument('--x-min', type=float, default=0.0,
                        help='Minimum binding energy in eV (default: 0)')
    parser.add_argument('--x-max', type=float, default=100.0,
                        help='Maximum binding energy in eV (default: 100)')
    parser.add_argument('--peak-placement', type=str, default='random',
                        choices=['random', 'uniform'],
                        help='Peak placement strategy (default: random)')
    parser.add_argument('--output', type=str, default='training_data/',
                        help='Output directory (default: training_data/)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--no-graph-only', action='store_true',
                        help='Skip clean graph visualization')
    parser.add_argument('--no-detailed', action='store_true',
                        help='Skip detailed analysis visualization')

    # Batch generation arguments
    parser.add_argument('--batch', action='store_true',
                        help='Generate a batch of datasets')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of datasets in batch (default: 10)')
    parser.add_argument('--peak-height-range', type=float, nargs=2, default=[80.0, 120.0],
                        help='Range of peak heights (default: 80 120)')
    parser.add_argument('--noise-level-range', type=float, nargs=2, default=[0.05, 0.15],
                        help='Range of noise levels (default: 0.05 0.15)')
    parser.add_argument('--save-batch-visualizations', action='store_true',
                        help='Save visualizations for each batch dataset')

    # Batch variations arguments
    parser.add_argument('--batch-variations', action='store_true',
                        help='Generate batch with peak count variations')
    parser.add_argument('--num-peaks-list', type=int, nargs='+', default=[3, 5, 7],
                        help='List of peak counts to generate (default: 3 5 7)')

    args = parser.parse_args()

    if args.batch or args.batch_variations:
        # Batch generation mode
        batch_gen = BatchDatasetGenerator(base_output_dir=args.output)

        if args.batch_variations:
            batch_gen.generate_batch_with_variations(
                batch_size=args.batch_size,
                num_points=args.num_points,
                num_peaks_list=args.num_peaks_list,
                peak_height=args.peak_height,
                noise_level=args.noise_level,
                x_min=args.x_min,
                x_max=args.x_max,
                save_visualizations=args.save_batch_visualizations,
                verbose=True,
            )
        else:
            batch_gen.generate_batch(
                batch_size=args.batch_size,
                num_points=args.num_points,
                num_peaks=args.num_peaks,
                peak_height_range=tuple(args.peak_height_range),
                noise_level_range=tuple(args.noise_level_range),
                x_min=args.x_min,
                x_max=args.x_max,
                peak_placement=args.peak_placement,
                save_visualizations=args.save_batch_visualizations,
                verbose=True,
            )
    else:
        # Single dataset generation mode
        generator = PhotoelectronSpectroscopyDataGenerator(
            num_points=args.num_points,
            num_peaks=args.num_peaks,
            peak_height=args.peak_height,
            noise_level=args.noise_level,
            x_min=args.x_min,
            x_max=args.x_max,
            seed=args.seed,
            peak_placement=args.peak_placement,
        )

        print("🔄 Generating photoelectron spectroscopy training data...")
        generator.generate()

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save files
        csv_path = output_dir / 'training_data.csv'
        json_path = output_dir / 'peak_ranges.json'
        graph_only_path = output_dir / 'graph_only.png'
        detailed_analysis_path = output_dir / 'detailed_analysis.png'

        print(f"\n💾 Saving files to {output_dir}...")
        generator.save_csv(str(csv_path))
        print(f"✓ CSV saved: {csv_path}")

        generator.save_peak_ranges(str(json_path))
        print(f"✓ Peak ranges saved: {json_path}")

        # Print summary
        generator.print_summary()

        # Generate visualizations
        if not args.no_graph_only:
            print("📊 Generating clean graph visualization...")
            generator.visualize_graph_only(str(graph_only_path))

        if not args.no_detailed:
            print("📊 Generating detailed analysis visualization...")
            generator.visualize_detailed_analysis(str(detailed_analysis_path))


if __name__ == '__main__':
    main()
