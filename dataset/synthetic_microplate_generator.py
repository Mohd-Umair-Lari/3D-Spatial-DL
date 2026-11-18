"""
Synthetic 3D Point Cloud Generator for Laboratory Microplates (Auto-Incremented & Multi-Sample)
Compliant with ANSI/SLAS 1-2004 through 4-2004 Microplate Standards.

Simulates high-resolution depth sensor scans with physical fluid meniscus curves,
variable sensor noise profiles, hardware stage mechanical tilt, and randomized
pipetting anomaly variations for precision laboratory automation QC.
Auto-numbers all generated files to avoid replacing previous scans.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


# Always resolve relative output paths from the repository root, not from the
# caller's current directory.  This prevents `dataset/dataset/...` when the
# script is launched from inside the dataset folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "synthetic_dataset"


def resolve_output_dir(output_dir: Optional[Union[str, os.PathLike]] = None) -> str:
    """Return one stable absolute output directory for every launch location."""
    if output_dir is None:
        return str(DEFAULT_OUTPUT_DIR)

    raw = Path(output_dir)
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        # CLI paths are project-relative, even when launched from dataset/.
        parts = list(raw.parts)
        while len(parts) >= 2 and parts[0] in (".", "dataset") and parts[1] == "dataset":
            parts.pop(0)
        if parts[:2] == ["dataset", "synthetic_dataset"]:
            resolved = (PROJECT_ROOT / Path(*parts)).resolve()
        elif parts[:1] == ["synthetic_dataset"]:
            resolved = (PROJECT_ROOT / "dataset" / Path(*parts)).resolve()
        else:
            resolved = (PROJECT_ROOT / raw).resolve()
    return str(resolved)


class MicroplateGeometry:
    """
    Standard geometric parameters based on ANSI/SLAS specifications.
    All dimensional units are in millimeters (mm).
    """
    PLATE_LENGTH = 127.76
    PLATE_WIDTH = 85.48
    PLATE_HEIGHT = 14.35

    # 96-Well Format Standard Specifications
    FMT_96 = {
        "rows": 8,
        "cols": 12,
        "pitch_x": 9.0,
        "pitch_y": 9.0,
        "offset_x": 14.38,
        "offset_y": 11.24,
        "well_radius": 3.4,       # ~6.8 mm diameter
        "well_depth": 10.8,       # mm from top surface
        "total_volume_uL": 360.0,
        "nominal_volume_uL": 50.0 # Standard working volume
    }

    # 12-Well Format Standard Specifications
    FMT_12 = {
        "rows": 3,
        "cols": 4,
        "pitch_x": 26.0,
        "pitch_y": 26.0,
        "offset_x": 24.88,
        "offset_y": 16.74,
        "well_radius": 11.05,     # ~22.1 mm diameter
        "well_depth": 17.5,
        "total_volume_uL": 6900.0,
        "nominal_volume_uL": 1000.0
    }


def find_next_available_index(output_dir: str, prefix: str = "scan_") -> int:
    """
    Scans output_dir for existing files matching prefix_XXX pattern and returns
    the next available integer index (1 if none exist) so previous files are never overwritten.
    """
    if not os.path.exists(output_dir):
        return 1

    max_idx = 0
    # Match patterns like scan_001_..., microplate_005_..., etc.
    pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
    for fname in os.listdir(output_dir):
        match = pattern.search(fname)
        if match:
            try:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
    return max_idx + 1


def resolve_auto_numbered_filepaths(
    output_ply: str,
    output_json: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    Resolves non-colliding numbered filepaths for a single sample generation.
    Example: 'microplate.ply' -> 'microplate_001.ply', 'microplate_002.ply', etc.
    """
    directory = os.path.dirname(os.path.abspath(output_ply))
    os.makedirs(directory, exist_ok=True)
    ply_name, ply_ext = os.path.splitext(os.path.basename(output_ply))

    if output_json:
        json_dir = os.path.dirname(os.path.abspath(output_json))
        os.makedirs(json_dir, exist_ok=True)
        json_name, json_ext = os.path.splitext(os.path.basename(output_json))
    else:
        json_dir = directory
        json_name = ply_name
        json_ext = ".json"

    counter = find_next_available_index(directory, prefix=f"{ply_name}_")

    while True:
        target_ply = os.path.join(directory, f"{ply_name}_{counter:03d}{ply_ext}")
        target_json = os.path.join(json_dir, f"{json_name}_{counter:03d}{json_ext}") if output_json else None

        ply_exists = os.path.exists(target_ply)
        json_exists = os.path.exists(target_json) if target_json else False

        if not ply_exists and not json_exists:
            return target_ply, target_json
        counter += 1


class SyntheticMicroplateGenerator:
    """
    Generates synthetic 3D point clouds (.ply) simulating optical triangulation,
    confocal chromatic, or structured-light depth sensors scanning microplates.
    Supports stochastic noise, hardware stage tilt, and randomized anomaly injections.
    """

    def __init__(
        self,
        format_type: str = "96",
        point_density_per_mm2: float = 6.0,
        sensor_noise_z_range: Tuple[float, float] = (0.015, 0.040),
        sensor_noise_xy_range: Tuple[float, float] = (0.005, 0.015),
        pipette_cv_percent: float = 1.5,
        simulate_stage_tilt: bool = True,
        seed: Optional[int] = None
    ):
        """
        Args:
            format_type: "96" or "12" well format.
            point_density_per_mm2: Spatial resolution (points per square mm).
            sensor_noise_z_range: Min and max Z Gaussian noise std dev (mm) [15-40 um].
            sensor_noise_xy_range: Min and max lateral XY optical jitter (mm).
            pipette_cv_percent: Liquid handler dispensing coefficient of variation (% CV).
            simulate_stage_tilt: Whether to simulate subtle robotic gantry/stage tilt.
            seed: Optional integer seed. When None, generates unique random instances every call.
        """
        self.format_type = format_type
        self.spec = MicroplateGeometry.FMT_96 if format_type == "96" else MicroplateGeometry.FMT_12
        self.point_density = point_density_per_mm2
        self.noise_z_range = sensor_noise_z_range
        self.noise_xy_range = sensor_noise_xy_range
        self.pipette_cv = pipette_cv_percent / 100.0
        self.simulate_stage_tilt = simulate_stage_tilt

        if seed is not None:
            np.random.seed(seed)

        self.wells = self._compute_well_centers()

    def _compute_well_centers(self) -> Dict[str, Tuple[float, float, int, int]]:
        """Computes Cartesian coordinates for all well centers."""
        wells = {}
        row_letters = "ABCDEFGH" if self.spec["rows"] == 8 else "ABC"
        for r_idx in range(self.spec["rows"]):
            for c_idx in range(self.spec["cols"]):
                well_id = f"{row_letters[r_idx]}{c_idx + 1:02d}"
                cx = self.spec["offset_x"] + c_idx * self.spec["pitch_x"]
                cy = self.spec["offset_y"] + r_idx * self.spec["pitch_y"]
                wells[well_id] = (cx, cy, r_idx, c_idx)
        return wells

    def _height_from_volume(self, volume_uL: float) -> float:
        """Converts fluid volume (uL = mm^3) to Z fluid surface height (mm)."""
        radius = self.spec["well_radius"]
        well_area = np.pi * (radius ** 2)
        fluid_column_height = volume_uL / well_area
        return float(-self.spec["well_depth"] + fluid_column_height)

    def _volume_from_height(self, z_surface: float) -> float:
        """Converts mean Z surface height back to volume in uL."""
        radius = self.spec["well_radius"]
        well_area = np.pi * (radius ** 2)
        fluid_column_height = max(0.0, z_surface - (-self.spec["well_depth"]))
        return float(fluid_column_height * well_area)

    def _apply_stage_tilt(self, x: np.ndarray, y: np.ndarray, tilt_angles: Tuple[float, float]) -> np.ndarray:
        """Applies subtle robotic plate stage tilt (yaw/pitch)."""
        center_x = MicroplateGeometry.PLATE_LENGTH / 2.0
        center_y = MicroplateGeometry.PLATE_WIDTH / 2.0
        tilt_x, tilt_y = tilt_angles
        return tilt_x * (x - center_x) + tilt_y * (y - center_y)

    def _generate_plate_surface(
        self,
        noise_z: float,
        noise_xy: float,
        tilt_angles: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates the upper flat plastic surface of the microplate with circular well cutouts."""
        length = MicroplateGeometry.PLATE_LENGTH
        width = MicroplateGeometry.PLATE_WIDTH
        num_points = int(length * width * self.point_density)

        x = np.random.uniform(0.0, length, num_points)
        y = np.random.uniform(0.0, width, num_points)
        z = np.zeros(num_points)

        radius = self.spec["well_radius"]
        mask_outside = np.ones(num_points, dtype=bool)
        for _, (cx, cy, _, _) in self.wells.items():
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            mask_outside &= (dist_sq >= radius ** 2)

        x_filtered = x[mask_outside] + np.random.normal(0, noise_xy, np.sum(mask_outside))
        y_filtered = y[mask_outside] + np.random.normal(0, noise_xy, np.sum(mask_outside))
        z_filtered = z[mask_outside] + np.random.normal(0, noise_z, np.sum(mask_outside))

        if self.simulate_stage_tilt:
            z_filtered += self._apply_stage_tilt(x_filtered, y_filtered, tilt_angles)

        points = np.column_stack([x_filtered, y_filtered, z_filtered])
        colors = np.tile(np.array([0.16, 0.17, 0.20]), (len(points), 1))
        return points, colors

    def _generate_well_fluid(
        self,
        cx: float,
        cy: float,
        z_base: float,
        meniscus_depth: float,
        noise_z: float,
        noise_xy: float,
        tilt_angles: Tuple[float, float],
        is_empty: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates 3D fluid surface points with realistic capillary meniscus physics.
        Model: z(r) = z_base + h_meniscus * (r / R)^2 (concave wetting meniscus).
        """
        radius = self.spec["well_radius"]
        well_area = np.pi * (radius ** 2)
        num_well_points = int(well_area * self.point_density * 1.3)

        pts_r = np.sqrt(np.random.uniform(0, radius ** 2, num_well_points))
        pts_theta = np.random.uniform(0, 2 * np.pi, num_well_points)

        x = cx + pts_r * np.cos(pts_theta)
        y = cy + pts_r * np.sin(pts_theta)

        if is_empty:
            z = np.full(num_well_points, -self.spec["well_depth"])
            color_rgb = np.array([0.08, 0.08, 0.10])
        else:
            norm_r = pts_r / radius
            z = z_base + meniscus_depth * (norm_r ** 2)
            color_rgb = np.array([0.15, 0.65, 0.88])

        x += np.random.normal(0, noise_xy, num_well_points)
        y += np.random.normal(0, noise_xy, num_well_points)
        z += np.random.normal(0, noise_z, num_well_points)

        if self.simulate_stage_tilt:
            z += self._apply_stage_tilt(x, y, tilt_angles)

        points = np.column_stack([x, y, z])
        colors = np.tile(color_rgb, (num_well_points, 1))
        ground_truth_z = np.full(num_well_points, z_base)
        return points, colors, ground_truth_z

    def _generate_spill_droplet(
        self,
        cx: float,
        cy: float,
        spill_radius: float,
        noise_z: float,
        noise_xy: float,
        tilt_angles: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulates an overfill spill pooling onto the top surface of the plate."""
        r_inner = self.spec["well_radius"]
        r_outer = spill_radius
        area_annulus = np.pi * (r_outer ** 2 - r_inner ** 2)
        num_spill_pts = int(area_annulus * self.point_density * 1.5)

        pts_r = np.sqrt(np.random.uniform(r_inner ** 2, r_outer ** 2, num_spill_pts))
        pts_theta = np.random.uniform(0, 2 * np.pi, num_spill_pts)

        x = cx + pts_r * np.cos(pts_theta)
        y = cy + pts_r * np.sin(pts_theta)

        norm_dist = (pts_r - r_inner) / (r_outer - r_inner)
        z = np.random.uniform(0.25, 0.45) * (1.0 - norm_dist ** 2)
        z += np.random.normal(0, noise_z, num_spill_pts)
        x += np.random.normal(0, noise_xy, num_spill_pts)
        y += np.random.normal(0, noise_xy, num_spill_pts)

        if self.simulate_stage_tilt:
            z += self._apply_stage_tilt(x, y, tilt_angles)

        points = np.column_stack([x, y, z])
        colors = np.tile(np.array([0.96, 0.50, 0.12]), (num_spill_pts, 1))
        return points, colors

    def generate_dataset(
        self,
        scenario: str = "baseline",
        output_ply_path: str = "scan.ply",
        output_json_path: Optional[str] = "microplate_metadata.json",
        randomize_anomalies: bool = True
    ) -> Dict[str, Any]:
        """
        Generates a synthetic point cloud with randomized noise and anomaly positions.
        """
        valid_scenarios = ["baseline", "underfill", "overfill_spill", "evaporation", "missing_well", "mixed"]
        if scenario not in valid_scenarios:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {valid_scenarios}")

        current_noise_z = float(np.random.uniform(*self.noise_z_range))
        current_noise_xy = float(np.random.uniform(*self.noise_xy_range))

        if self.simulate_stage_tilt:
            tilt_angles = (float(np.random.normal(0, 0.0006)), float(np.random.normal(0, 0.0006)))
        else:
            tilt_angles = (0.0, 0.0)

        all_points = []
        all_colors = []
        metadata_wells: Dict[str, Dict[str, Any]] = {}

        # 1. Base plate upper surface
        plate_pts, plate_colors = self._generate_plate_surface(current_noise_z, current_noise_xy, tilt_angles)
        all_points.append(plate_pts)
        all_colors.append(plate_colors)

        # 2. Configure anomaly wells dynamically
        well_ids = list(self.wells.keys())
        underfill_targets = []
        overfill_targets = []
        missing_targets = []

        if randomize_anomalies:
            if scenario == "underfill":
                k = int(np.random.choice([1, 2, 3]))
                underfill_targets = [str(x) for x in np.random.choice(well_ids, size=k, replace=False)]
            elif scenario == "overfill_spill":
                overfill_targets = [str(x) for x in np.random.choice(well_ids, size=1, replace=False)]
            elif scenario == "missing_well":
                k = int(np.random.choice([1, 2]))
                missing_targets = [str(x) for x in np.random.choice(well_ids, size=k, replace=False)]
            elif scenario == "mixed":
                chosen = [str(x) for x in np.random.choice(well_ids, size=4, replace=False)]
                underfill_targets = [chosen[0], chosen[1]]
                overfill_targets = [chosen[2]]
                missing_targets = [chosen[3]]
        else:
            if scenario in ["underfill", "mixed"]:
                underfill_targets = ["C04", "F09"]
            if scenario in ["overfill_spill", "mixed"]:
                overfill_targets = ["D06"]
            if scenario in ["missing_well", "mixed"]:
                missing_targets = ["B03"]

        nominal_vol = self.spec["nominal_volume_uL"]
        center_x = MicroplateGeometry.PLATE_LENGTH / 2.0
        center_y = MicroplateGeometry.PLATE_WIDTH / 2.0
        max_dist_from_center = np.sqrt(center_x ** 2 + center_y ** 2)

        for well_id, (cx, cy, r_idx, c_idx) in self.wells.items():
            well_nominal_vol = nominal_vol * (1.0 + np.random.normal(0, self.pipette_cv))
            current_vol = well_nominal_vol
            current_z = self._height_from_volume(current_vol)
            status = "normal"
            is_empty = False
            has_spill = False
            meniscus_depth = float(np.random.uniform(0.28, 0.42))

            if well_id in underfill_targets:
                severity = np.random.uniform(0.08, 0.35)
                current_vol = nominal_vol * severity
                current_z = self._height_from_volume(current_vol)
                status = "anomaly_underfill"

            elif well_id in overfill_targets:
                current_vol = self.spec["total_volume_uL"] * np.random.uniform(1.04, 1.12)
                current_z = float(np.random.uniform(0.18, 0.32))
                status = "anomaly_overfill_spill"
                has_spill = True

            elif scenario == "evaporation":
                dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
                gradient_factor = (dist / max_dist_from_center) ** np.random.uniform(1.6, 2.0)
                loss_fraction = np.random.uniform(0.35, 0.55) * gradient_factor
                current_vol = nominal_vol * (1.0 - loss_fraction)
                current_z = self._height_from_volume(current_vol)
                status = "evaporation_gradient" if gradient_factor > 0.35 else "normal"

            elif well_id in missing_targets:
                current_vol = 0.0
                current_z = -self.spec["well_depth"]
                status = "anomaly_empty"
                is_empty = True

            fluid_pts, fluid_colors, _ = self._generate_well_fluid(
                cx, cy, current_z,
                meniscus_depth=meniscus_depth,
                noise_z=current_noise_z,
                noise_xy=current_noise_xy,
                tilt_angles=tilt_angles,
                is_empty=is_empty
            )
            all_points.append(fluid_pts)
            all_colors.append(fluid_colors)

            if has_spill:
                spill_radius = self.spec["well_radius"] + float(np.random.uniform(2.5, 4.5))
                spill_pts, spill_colors = self._generate_spill_droplet(
                    cx, cy,
                    spill_radius=spill_radius,
                    noise_z=current_noise_z,
                    noise_xy=current_noise_xy,
                    tilt_angles=tilt_angles
                )
                all_points.append(spill_pts)
                all_colors.append(spill_colors)

            metadata_wells[well_id] = {
                "well_id": well_id,
                "row": r_idx,
                "column": c_idx,
                "center_xyz_mm": [float(cx), float(cy), float(current_z)],
                "target_volume_uL": float(round(current_vol, 3)),
                "nominal_volume_uL": float(round(nominal_vol, 3)),
                "volume_delta_uL": float(round(current_vol - nominal_vol, 3)),
                "status": status
            }

        pcd_points = np.vstack(all_points)
        pcd_colors = np.vstack(all_colors)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd_points)
        point_cloud.colors = o3d.utility.Vector3dVector(pcd_colors)

        os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
        o3d.io.write_point_cloud(output_ply_path, point_cloud, write_ascii=False)

        payload = {
            "format": f"{self.format_type}-well",
            "scenario": scenario,
            "total_points": len(pcd_points),
            "sensor_noise_z_um": round(current_noise_z * 1000.0, 2),
            "sensor_noise_xy_um": round(current_noise_xy * 1000.0, 2),
            "stage_tilt_rad": [float(tilt_angles[0]), float(tilt_angles[1])],
            "wells_count": len(metadata_wells),
            "anomalies": {
                "underfill": underfill_targets,
                "overfill_spill": overfill_targets,
                "missing": missing_targets
            },
            "wells": metadata_wells
        }

        if output_json_path:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        return payload

    def generate_batch(
        self,
        num_samples: int = 50,
        scenarios: Union[str, List[str]] = "all",
        output_dir: Optional[str] = None,
        prefix: str = "scan_"
    ) -> List[Dict[str, Any]]:
        """
        Loops generation up to num_samples files, sequentially numbering each file
        and detecting existing files so previous scans are NEVER replaced or overwritten.

        Args:
            num_samples: Total number of point cloud datasets to generate in the loop (default: 50).
            scenarios: "all" (cycles through 6 scenarios), "random", or specific scenario name.
            output_dir: Directory where .ply, .json, and manifest.json will be saved.
            prefix: Filename prefix for scans (e.g. "scan_").
        """
        output_dir = resolve_output_dir(output_dir)
        all_scenarios = ["baseline", "underfill", "overfill_spill", "evaporation", "missing_well", "mixed"]
        os.makedirs(output_dir, exist_ok=True)

        # Detect highest existing index so we append seamlessly without replacing
        start_idx = find_next_available_index(output_dir, prefix=prefix)
        end_idx = start_idx + num_samples - 1

        print(f"\n[Batch Generator] Starting loop for {num_samples} files (indices {start_idx:03d} to {end_idx:03d}) in '{output_dir}'...")

        catalog = []
        for step, file_idx in enumerate(range(start_idx, start_idx + num_samples), start=1):
            if scenarios == "all":
                scenario = all_scenarios[(file_idx - 1) % len(all_scenarios)]
            elif scenarios == "random":
                scenario = str(np.random.choice(all_scenarios))
            elif isinstance(scenarios, list):
                scenario = str(np.random.choice(scenarios))
            else:
                scenario = scenarios

            # Keep the baseline files compactly numbered.  The scenario is
            # still preserved in the JSON metadata and manifest, so removing
            # it from the filename does not remove provenance.
            scenario_suffix = "" if scenario == "baseline" else f"_{scenario}"
            ply_filename = f"{prefix}{file_idx:03d}{scenario_suffix}.ply"
            json_filename = f"{prefix}{file_idx:03d}{scenario_suffix}.json"

            ply_path = os.path.join(output_dir, ply_filename)
            json_path = os.path.join(output_dir, json_filename)

            meta = self.generate_dataset(
                scenario=scenario,
                output_ply_path=ply_path,
                output_json_path=json_path,
                randomize_anomalies=True
            )

            entry = {
                "sample_id": file_idx,
                "file_ply": ply_filename,
                "file_json": json_filename,
                "scenario": scenario,
                "points": meta["total_points"],
                "sensor_noise_z_um": meta["sensor_noise_z_um"],
                "anomalies": meta["anomalies"]
            }
            catalog.append(entry)

            anomaly_summary = []
            for k, v in meta["anomalies"].items():
                if v:
                    anomaly_summary.append(f"{k}:{v}")
            anomaly_str = ", ".join(anomaly_summary) if anomaly_summary else "none"

            print(f" -> [{step:02d}/{num_samples:02d}] #{file_idx:03d} | {ply_filename:30s} | Noise: {meta['sensor_noise_z_um']:4.1f} um | Anomaly: {anomaly_str}")

        # Update or create dataset_manifest.json (preserving previous history)
        manifest_path = os.path.join(output_dir, "dataset_manifest.json")
        existing_samples = []
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    old_manifest = json.load(f)
                    existing_samples = old_manifest.get("samples", [])
            except Exception:
                existing_samples = []

        combined_samples = existing_samples + catalog
        manifest_data = {
            "total_generated_scans": len(combined_samples),
            "format": f"{self.format_type}-well",
            "samples": combined_samples
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        print(f"\n[Loop Complete] Successfully generated {num_samples} files.")
        print(f"Master manifest updated: {manifest_path} (Total tracked scans: {len(combined_samples)})\n")
        return catalog


def verify_point_cloud_analytics(ply_path: str, json_path: str):
    """
    Demonstrates spatial segmentation via KDTree and DBSCAN clustering,
    verifying compatibility for downstream FastAPI & LangChain ingestion.
    """
    print(f"\n[Verification] Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(f"[Verification] Total points loaded: {len(points):,}")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 1. Scipy KDTree radius search
    kdtree = KDTree(points)
    test_wells = meta.get("anomalies", {}).get("underfill", [])
    if not test_wells:
        test_wells = ["D06"]
    query_well = test_wells[0]

    if query_well in meta["wells"]:
        well_info = meta["wells"][query_well]
        query_pt = np.array(well_info["center_xyz_mm"])
        indices = kdtree.query_ball_point(query_pt, r=3.0)
        print(f"[KDTree Query] Found {len(indices)} points near {query_well} (status: {well_info['status']})")

    # 2. Open3D DBSCAN clustering
    fluid_mask = points[:, 2] < -0.1
    fluid_pcd = pcd.select_by_index(np.where(fluid_mask)[0])
    labels = np.array(fluid_pcd.cluster_dbscan(eps=1.5, min_points=15))
    num_clusters = labels.max() + 1
    print(f"[DBSCAN Clustering] Identified {num_clusters} distinct fluid well clusters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Microplate 3D Point Cloud Generator (Loop up to 50+ files)")
    parser.add_argument("--format", type=str, default="96", choices=["96", "12"], help="Microplate format (96 or 12)")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "random", "baseline", "underfill", "overfill_spill", "evaporation", "missing_well", "mixed"],
        help="QC Scenario variation (or 'all'/'random' for batch loop)"
    )
    parser.add_argument("--density", type=float, default=6.0, help="Point density per mm^2")
    parser.add_argument("--count", "--batch", dest="count", type=int, default=50, help="Number of files to loop and generate (default: 50)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (project-relative; defaults to dataset/synthetic_dataset)")
    parser.add_argument("--prefix", type=str, default="scan_", help="Filename prefix for numbered files")
    parser.add_argument("--verify", action="store_true", help="Run KDTree and DBSCAN verification check on last file")

    args = parser.parse_args()

    generator = SyntheticMicroplateGenerator(
        format_type=args.format,
        point_density_per_mm2=args.density,
        sensor_noise_z_range=(0.015, 0.040),
        sensor_noise_xy_range=(0.005, 0.015),
        pipette_cv_percent=1.5,
        simulate_stage_tilt=True
    )

    # Run the loop to generate up to 50 (or requested count) numbered files
    results = generator.generate_batch(
        num_samples=args.count,
        scenarios=args.scenario,
        output_dir=args.output_dir,
        prefix=args.prefix
    )

    if args.verify and results:
        output_dir = resolve_output_dir(args.output_dir)
        last_ply = os.path.join(output_dir, results[-1]["file_ply"])
        last_json = os.path.join(output_dir, results[-1]["file_json"])
        verify_point_cloud_analytics(last_ply, last_json)
