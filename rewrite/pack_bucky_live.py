#!/usr/bin/env python3
import argparse
import csv
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy import ndimage


DEFAULT_MESH = Path("/Users/dofa/Desktop/high_poly_bucky.obj")


@dataclass
class Sphere:
    center: np.ndarray
    radius: float


class StopRequested(Exception):
    pass


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def install_signal_handlers(stop_flag):
    def request_stop(_signum, _frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast live sphere packing for a mesh.")
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--output", type=Path, default=Path("outputs/high_poly_bucky_spheres.json"))
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing output JSON.")
    parser.add_argument("--no-gui", action="store_true", help="Run headless.")
    parser.add_argument("--max-spheres", type=int, default=0, help="Maximum spheres to pack. Use 0 to run until stopped.")
    parser.add_argument("--sample-count", type=int, default=140000)
    parser.add_argument("--replenish-count", type=int, default=35000)
    parser.add_argument("--replenish-every", type=int, default=50)
    parser.add_argument("--voxel-resolution", type=int, default=170)
    parser.add_argument("--target-triangles", type=int, default=70000)
    parser.add_argument("--sample-batch", type=int, default=70000)
    parser.add_argument("--update-chunk", type=int, default=65536)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--display-every", type=int, default=1)
    parser.add_argument("--sphere-resolution", type=int, default=10)
    parser.add_argument("--min-radius", type=float, default=None)
    parser.add_argument("--min-radius-frac", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def log(message: str):
    print(message, flush=True)


def load_mesh(mesh_path: Path, target_triangles: int):
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    log(f"loading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    if mesh.is_empty() or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"Open3D could not load a triangle mesh from {mesh_path}")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    if len(mesh.triangles) > target_triangles:
        log(f"decimating render/query mesh: {len(mesh.triangles)} -> {target_triangles} triangles")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        mesh.remove_degenerate_triangles()

    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    bbox = mesh.get_axis_aligned_bounding_box()
    bbox_min = np.asarray(bbox.min_bound, dtype=np.float32)
    bbox_max = np.asarray(bbox.max_bound, dtype=np.float32)
    diag = float(np.linalg.norm(bbox_max - bbox_min))

    log(f"mesh ready: {len(vertices):,} vertices, {len(triangles):,} triangles, bbox diagonal {diag:.4f}")
    if tri_mesh.is_watertight and tri_mesh.volume > 0:
        log(f"mesh volume: {tri_mesh.volume:.4f}")
    else:
        log("mesh is not watertight; volume/fill percentage will be skipped")

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return mesh, scene, bbox_min, bbox_max, diag, tri_mesh


def mesh_occupancy_and_distance(scene, points: np.ndarray):
    points = np.asarray(points, dtype=np.float32)
    query = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    occupancy = scene.compute_occupancy(query).numpy() > 0.5
    distance = scene.compute_distance(query).numpy().astype(np.float32)
    return occupancy, distance


class VoxelInterior:
    def __init__(self, tri_mesh, diagonal, resolution, min_radius):
        pitch = diagonal / float(resolution)
        log(f"building closed voxel interior: resolution={resolution}, pitch={pitch:.5f}")
        voxels = tri_mesh.voxelized(pitch)
        surface = voxels.matrix.astype(bool)
        filled = ndimage.binary_fill_holes(surface)
        distance = ndimage.distance_transform_edt(filled) * float(np.max(voxels.pitch))

        # Keep centers safely away from the voxel shell. The half-voxel margin is
        # what prevents spheres from poking through the displayed wireframe.
        clearance = distance - (math.sqrt(3.0) * float(np.max(voxels.pitch)) * 0.5)
        viable = filled & (clearance > min_radius)
        indices = np.argwhere(viable)
        if len(indices) == 0:
            raise RuntimeError("Voxel solid has no viable interior cells. Try lowering --min-radius-frac or raising --voxel-resolution.")

        points = voxels.indices_to_points(indices).astype(np.float32)
        radii = clearance[viable].astype(np.float32)
        self.points = points
        self.radii = radii
        self.pitch = float(np.max(voxels.pitch))
        self.shape = filled.shape
        log(
            "voxel solid ready: "
            f"shape={filled.shape}, surface={int(surface.sum()):,}, "
            f"solid={int(filled.sum()):,}, viable={len(points):,}, "
            f"max interior radius={float(radii.max()):.5f}"
        )

    def sample(self, count, rng):
        if count <= 0 or count >= len(self.points):
            return self.points.copy(), self.radii.copy()
        choice = rng.choice(len(self.points), size=count, replace=False)
        return self.points[choice].copy(), self.radii[choice].copy()


def load_spheres(path: Path):
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    spheres = []
    for item in data:
        center = np.asarray(item["center"], dtype=np.float32)
        radius = float(item["radius"])
        if center.shape == (3,) and radius > 0:
            spheres.append(Sphere(center=center, radius=radius))
    return spheres


def save_spheres(path: Path, spheres, tri_mesh):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "center": [float(v) for v in sphere.center],
            "radius": float(sphere.radius),
        }
        for sphere in spheres
    ]
    path.write_text(json.dumps(payload, indent=2))

    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "radius"])
        for sphere in spheres:
            writer.writerow([float(sphere.center[0]), float(sphere.center[1]), float(sphere.center[2]), float(sphere.radius)])

    total_volume = sum((4.0 / 3.0) * math.pi * sphere.radius ** 3 for sphere in spheres)
    fill = None
    if tri_mesh.is_watertight and tri_mesh.volume > 0:
        fill = total_volume / tri_mesh.volume
    return total_volume, fill


def unique_wire_edges(triangles: np.ndarray):
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def build_wireframe(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    edges = unique_wire_edges(triangles)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.55, 0.62, 0.70]]), (len(edges), 1)))
    return line_set


def sphere_color(radius, min_radius, max_radius):
    if max_radius <= min_radius:
        return [0.95, 0.25, 0.18]
    t = (math.log(radius) - math.log(min_radius)) / (math.log(max_radius) - math.log(min_radius))
    t = max(0.0, min(1.0, t))
    return [0.10 + 0.85 * t, 0.45 + 0.35 * (1.0 - t), 0.95 - 0.75 * t]


def add_sphere_geometry(visualizer, sphere, min_radius, max_radius, resolution):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=sphere.radius, resolution=resolution)
    mesh.translate(sphere.center)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(sphere_color(sphere.radius, min_radius, max_radius))
    visualizer.add_geometry(mesh, reset_bounding_box=False)


def pump_gui(visualizer, frames=1, delay=0.0):
    for _ in range(frames):
        if not visualizer.poll_events():
            return False
        visualizer.update_renderer()
        if delay > 0:
            time.sleep(delay)
    return True


def create_visualizer(mesh, spheres, min_radius, max_radius, resolution):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="SpherePacker Better - high_poly_bucky.obj", width=1280, height=900)
    visualizer.add_geometry(build_wireframe(mesh))
    for idx, sphere in enumerate(spheres, start=1):
        add_sphere_geometry(visualizer, sphere, min_radius, max_radius, resolution)
        if idx % 100 == 0:
            pump_gui(visualizer, frames=1)
    render = visualizer.get_render_option()
    render.background_color = np.asarray([0.03, 0.035, 0.045])
    render.line_width = 1.0
    render.mesh_show_back_face = True
    pump_gui(visualizer, frames=5, delay=0.01)
    return visualizer


def append_samples(points_t, clearance_t, interior, count, rng, device):
    points, clearance = interior.sample(count, rng)
    new_points_t = torch.as_tensor(points, dtype=torch.float32, device=device)
    new_clearance_t = torch.as_tensor(clearance, dtype=torch.float32, device=device)
    if points_t is None:
        return new_points_t, new_clearance_t
    return torch.cat((points_t, new_points_t), dim=0), torch.cat((clearance_t, new_clearance_t), dim=0)


def apply_existing_sphere_clearance(points_t, clearance_t, sphere, chunk_size):
    center_t = torch.as_tensor(sphere.center, dtype=torch.float32, device=points_t.device)
    radius = torch.tensor(float(sphere.radius), dtype=torch.float32, device=points_t.device)
    for start in range(0, points_t.shape[0], chunk_size):
        end = min(start + chunk_size, points_t.shape[0])
        dist_to_sphere = torch.linalg.norm(points_t[start:end] - center_t, dim=1) - radius
        clearance_t[start:end] = torch.minimum(clearance_t[start:end], dist_to_sphere)


def prune_dead_samples(points_t, clearance_t, min_radius, keep_floor=20000):
    live = clearance_t > min_radius
    live_count = int(live.sum().detach().cpu().item())
    if live_count < keep_floor:
        return points_t, clearance_t, live_count
    return points_t[live], clearance_t[live], live_count


def pack(args):
    stop_flag = {"stop": False}
    install_signal_handlers(stop_flag)

    rng = np.random.default_rng(args.seed)
    device = choose_device()
    log(f"compute device: {device}")

    mesh, scene, bbox_min, bbox_max, diag, tri_mesh = load_mesh(args.mesh, args.target_triangles)
    min_radius = args.min_radius if args.min_radius is not None else diag * args.min_radius_frac
    log(f"minimum sphere radius: {min_radius:.6f}")

    args.output = args.output.resolve()
    if args.fresh and args.output.exists():
        args.output.unlink()

    spheres = load_spheres(args.output)
    if spheres:
        log(f"resuming {len(spheres):,} spheres from {args.output}")
    else:
        log("starting a fresh packing")

    interior = VoxelInterior(
        tri_mesh=tri_mesh,
        diagonal=diag,
        resolution=args.voxel_resolution,
        min_radius=min_radius,
    )
    sample_points, sample_clearance = interior.sample(args.sample_count, rng)
    log(f"using {len(sample_points):,} voxel-solid samples")
    points_t = torch.as_tensor(sample_points, dtype=torch.float32, device=device)
    clearance_t = torch.as_tensor(sample_clearance, dtype=torch.float32, device=device)

    for idx, sphere in enumerate(spheres, start=1):
        apply_existing_sphere_clearance(points_t, clearance_t, sphere, args.update_chunk)
        if idx % 100 == 0:
            log(f"rebuilt clearance from saved spheres: {idx:,}/{len(spheres):,}")

    min_seen_radius = min([sphere.radius for sphere in spheres], default=min_radius)
    max_seen_radius = max([sphere.radius for sphere in spheres], default=min_radius * 2)
    visualizer = None if args.no_gui else create_visualizer(
        mesh=mesh,
        spheres=spheres,
        min_radius=min_seen_radius,
        max_radius=max_seen_radius,
        resolution=args.sphere_resolution,
    )

    start_time = time.time()
    last_save = time.time()
    try:
        while args.max_spheres <= 0 or len(spheres) < args.max_spheres:
            if stop_flag["stop"]:
                raise StopRequested()

            best_radius_t, best_idx_t = torch.max(clearance_t, dim=0)
            best_radius = float(best_radius_t.detach().cpu().item())
            if best_radius <= min_radius:
                log("sample cloud is exhausted; replenishing interior samples")
                points_t, clearance_t = append_samples(
                points_t,
                clearance_t,
                interior,
                args.replenish_count,
                rng,
                device,
            )
                for sphere in spheres:
                    apply_existing_sphere_clearance(points_t[-args.replenish_count :], clearance_t[-args.replenish_count :], sphere, args.update_chunk)
                best_radius_t, best_idx_t = torch.max(clearance_t, dim=0)
                best_radius = float(best_radius_t.detach().cpu().item())
                if best_radius <= min_radius:
                    log("no viable sample remains above the minimum radius")
                    break

            best_idx = int(best_idx_t.detach().cpu().item())
            center = points_t[best_idx].detach().cpu().numpy().astype(np.float32)
            sphere = Sphere(center=center, radius=best_radius)
            spheres.append(sphere)

            apply_existing_sphere_clearance(points_t, clearance_t, sphere, args.update_chunk)

            if len(spheres) % args.replenish_every == 0:
                points_t, clearance_t, live_count = prune_dead_samples(points_t, clearance_t, min_radius)
                log(f"live samples after pruning: {live_count:,}")
                points_t, clearance_t = append_samples(
                points_t,
                clearance_t,
                interior,
                args.replenish_count,
                rng,
                device,
            )
                tail_points = points_t[-args.replenish_count :]
                tail_clearance = clearance_t[-args.replenish_count :]
                for old_sphere in spheres:
                    apply_existing_sphere_clearance(tail_points, tail_clearance, old_sphere, args.update_chunk)

            min_seen_radius = min(min_seen_radius, sphere.radius)
            max_seen_radius = max(max_seen_radius, sphere.radius)

            if visualizer is not None and len(spheres) % args.display_every == 0:
                add_sphere_geometry(visualizer, sphere, min_seen_radius, max_seen_radius, args.sphere_resolution)
                if not pump_gui(visualizer, frames=3, delay=0.002):
                    raise StopRequested()

            if len(spheres) % args.save_every == 0 or time.time() - last_save > 20:
                total_volume, fill = save_spheres(args.output, spheres, tri_mesh)
                last_save = time.time()
                fill_text = f", fill {fill * 100:.2f}%" if fill is not None else ""
                log(f"saved {len(spheres):,} spheres, last r={sphere.radius:.5f}, volume {total_volume:.4f}{fill_text}")
            else:
                log(f"sphere {len(spheres):,}: r={sphere.radius:.5f}, center=({sphere.center[0]:.4f}, {sphere.center[1]:.4f}, {sphere.center[2]:.4f})")

    except StopRequested:
        log("stop requested; saving current packing")
    finally:
        total_volume, fill = save_spheres(args.output, spheres, tri_mesh)
        elapsed = time.time() - start_time
        fill_text = f", fill {fill * 100:.2f}%" if fill is not None else ""
        log(f"done: {len(spheres):,} spheres in {elapsed:.1f}s, sphere volume {total_volume:.4f}{fill_text}")
        log(f"saved: {args.output}")
        log(f"saved: {args.output.with_suffix('.csv')}")
        if visualizer is not None:
            log("leave the window open to inspect; close it when done")
            while pump_gui(visualizer, frames=1):
                time.sleep(0.03)


def main():
    args = parse_args()
    try:
        pack(args)
    except Exception as exc:
        log(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
