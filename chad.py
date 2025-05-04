import time
import json
import os
import numpy as np
import torch
import trimesh
import open3d as o3d
import colorsys


# standard stuff mps better than cuda
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
useHalf = device == "cuda"

#read the github pages post for this
def gpuContains(points, meshGpu, rayDirection=None, epsilon=1e-8):
    with torch.no_grad():
        vertices, faces = meshGpu
        if rayDirection is None:
            rayDirection = torch.tensor([1.0, 0.0, 0.0], device=device)
        elif not isinstance(rayDirection, torch.Tensor):
            rayDirection = torch.tensor(rayDirection, device=device)
        d = rayDirection / torch.norm(rayDirection)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = torch.cross(d.expand_as(edge2), edge2, dim=1)
        a = torch.sum(edge1 * h, dim=1)
        validTriangle = torch.abs(a) > epsilon
        f = torch.zeros_like(a)
        f[validTriangle] = 1.0 / a[validTriangle]
        n = points.shape[0]
        fCount = faces.shape[0]
        intersections = torch.zeros(n, device=device, dtype=torch.int32)
        batchSize = min(10000, n)
        for start in range(0, n, batchSize):
            end = min(start + batchSize, n)
            pBatch = points[start:end]
            s = pBatch.unsqueeze(1) - v0.unsqueeze(0)
            u = f.unsqueeze(0) * torch.sum(s * h.unsqueeze(0), dim=2)
            condU = (u < 0) | (u > 1)
            q = torch.cross(s, edge1.unsqueeze(0), dim=2)
            vVal = f.unsqueeze(0) * torch.sum(d.unsqueeze(0).unsqueeze(0) * q, dim=2)
            condV = (vVal < 0) | ((u + vVal) > 1)
            t = f.unsqueeze(0) * torch.sum(edge2.unsqueeze(0) * q, dim=2)
            condT = t <= epsilon
            condValid = (~validTriangle).unsqueeze(0).expand(pBatch.shape[0], fCount)
            intersect = (~condU) & (~condV) & (~condT) & (~condValid)
            intersections[start:end] = torch.sum(intersect, dim=1)
        inside = (intersections % 2 == 1)
        return inside

def gpuNearestNeighbor(points, referencePoints):
    with torch.no_grad():
        batchSize = 10000
        minDists = torch.empty(points.shape[0], device=device)
        for i in range(0, points.shape[0], batchSize):
            endIdx = min(i + batchSize, points.shape[0])
            batchPoints = points[i:endIdx]
            dists = torch.cdist(batchPoints, referencePoints)
            minDists[i:endIdx], _ = torch.min(dists, dim=1)
        return minDists

def logMap(value, minVal, maxVal):
    if minVal <= 0 or maxVal <= 0:
        raise ValueError("minVal and maxVal must be positive")
    logMin = np.log(minVal)
    logMax = np.log(maxVal)
    logValue = np.log(value)
    return (logValue - logMin) / (logMax - logMin)

def getColorFromRadius(radius, minRadius, maxRadius):
    if minRadius == maxRadius:
        return [1.0, 0.0, 0.0]
    try:
        normalized = logMap(radius, minRadius, maxRadius)
        hue = normalized * 300
        rgb = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
        return list(rgb)
    except ValueError as e:
        print(f"error in getColorFromRadius: {e}. using fallback color.")
        return [1.0, 1.0, 1.0]

def findLargestSphereGlobal(meshGpu, surfacePoints, globalBounds, gridSize=15):
    with torch.no_grad():
        dtype = surfacePoints.dtype
        xVals = np.linspace(globalBounds[0][0], globalBounds[0][1], gridSize)
        yVals = np.linspace(globalBounds[1][0], globalBounds[1][1], gridSize)
        zVals = np.linspace(globalBounds[2][0], globalBounds[2][1], gridSize)
        gridPoints = []
        for x in xVals:
            for y in yVals:
                for z in zVals:
                    gridPoints.append([x, y, z])
        gridPoints = torch.tensor(gridPoints, device=device, dtype=dtype)
        insideMask = gpuContains(gridPoints, meshGpu)
        insidePoints = gridPoints[insideMask]
        if insidePoints.shape[0] == 0:
            return None, None
        distances = gpuNearestNeighbor(insidePoints, surfacePoints)
        maxIdx = torch.argmax(distances)
        bestRadius = distances[maxIdx].item()
        bestCenter = insidePoints[maxIdx].cpu().numpy()
        return bestCenter, bestRadius

def findTangentSpheresBatch(existingSpheres, surfacePoints, meshGpu, globalBounds, numCandidates=100, minRadius=1e-8):
    with torch.no_grad():
        dtype = torch.float16 if useHalf else torch.float32
        if existingSpheres is None or len(existingSpheres) == 0:
            center, radius = findLargestSphereGlobal(meshGpu, surfacePoints, globalBounds)
            if center is not None and radius > minRadius:
                return [(center, radius)]
            return []
        centers = existingSpheres[:, :3]
        radii = existingSpheres[:, 3]
        weights = radii**3
        weights = weights / weights.sum()
        numAnchors = min(20, len(existingSpheres))
        indices = torch.multinomial(weights, numAnchors, replacement=False)
        anchorCenters = centers[indices]
        anchorRadii = radii[indices]
        directionsPerAnchor = numCandidates // numAnchors
        directions = torch.randn(numAnchors, directionsPerAnchor, 3, device=device, dtype=dtype)
        directions = directions / torch.norm(directions, dim=2, keepdim=True)
        anchorCentersExp = anchorCenters.unsqueeze(1).expand(-1, directionsPerAnchor, -1)
        anchorRadiiExp = anchorRadii.unsqueeze(1).expand(-1, directionsPerAnchor)
        dMax = torch.full((numAnchors, directionsPerAnchor), float("inf"), device=device, dtype=dtype)
        for j in range(3):
            dComp = directions[:, :, j]
            comp = anchorCentersExp[:, :, j]
            posMask = dComp > 0
            negMask = dComp < 0
            dJ = torch.empty_like(dMax)
            if posMask.any():
                dJ[posMask] = (globalBounds[j][1] - comp[posMask]) / dComp[posMask]
            if negMask.any():
                dJ[negMask] = (globalBounds[j][0] - comp[negMask]) / dComp[negMask]
            dMax = torch.minimum(dMax, dJ)
        low = anchorRadiiExp.clone()
        high = dMax.clone()
        for _ in range(20):
            mid = (low + high) / 2.0
            candidateCenters = anchorCentersExp + mid.unsqueeze(-1) * directions
            candidateRadii = mid - anchorRadiiExp
            flatCenters = candidateCenters.reshape(-1, 3)
            flatRadii = candidateRadii.reshape(-1)
            insideMask = gpuContains(flatCenters, meshGpu)
            surfaceDists = gpuNearestNeighbor(flatCenters, surfacePoints)
            validSurface = flatRadii <= surfaceDists
            allCentersExp = flatCenters.unsqueeze(1).expand(-1, centers.shape[0], -1)
            allRadiiExp = flatRadii.unsqueeze(1).expand(-1, radii.shape[0])
            centerDists = torch.norm(allCentersExp - centers.unsqueeze(0), dim=2)
            validOverlap = (centerDists >= allRadiiExp + radii.unsqueeze(0) - 1e-6).all(dim=1)
            validMask = insideMask & validSurface & validOverlap
            validMask = validMask.reshape(numAnchors, directionsPerAnchor)
            low = torch.where(validMask, mid, low)
            high = torch.where(~validMask, mid, high)
            if torch.max(high - low) < 1e-6 * torch.max(low):
                break
        candidateCenters = anchorCentersExp + low.unsqueeze(-1) * directions
        candidateRadii = low - anchorRadiiExp
        candidateCenters = candidateCenters.reshape(-1, 3)
        candidateRadii = candidateRadii.reshape(-1)
        insideMask = gpuContains(candidateCenters, meshGpu)
        surfaceDists = gpuNearestNeighbor(candidateCenters, surfacePoints)
        validSurface = candidateRadii <= surfaceDists
        validOverlap = torch.ones_like(insideMask, dtype=torch.bool)
        if centers.shape[0] > 0:
            allCentersExp = candidateCenters.unsqueeze(1).expand(-1, centers.shape[0], -1)
            allRadiiExp = candidateRadii.unsqueeze(1).expand(-1, radii.shape[0])
            centerDists = torch.norm(allCentersExp - centers.unsqueeze(0), dim=2)
            validOverlap = (centerDists >= allRadiiExp + radii.unsqueeze(0) - 1e-6).all(dim=1)
        validMask = insideMask & validSurface & validOverlap & (candidateRadii > minRadius)
        validCenters = candidateCenters[validMask]
        validRadii = candidateRadii[validMask]
        if len(validRadii) > 0:
            sortedIndices = torch.argsort(validRadii, descending=True)
            topK = min(5, len(sortedIndices))
            return [(validCenters[idx].cpu().numpy(), validRadii[idx].item())
                    for idx in sortedIndices[:topK]]
        return []

overallStartTime = time.time()
meshFile = "objects/dragon.obj"
jsonFilename = "dic.json"
print(f"loading mesh from {meshFile}...")
meshO3d = o3d.io.read_triangle_mesh(meshFile)
if not meshO3d.has_triangle_normals():
    meshO3d.compute_triangle_normals()
meshO3d.compute_vertex_normals()
targetTriangles = 50000
if len(meshO3d.triangles) > targetTriangles:
    print(f"simplifying mesh from {len(meshO3d.triangles)} to {targetTriangles} triangles...")
    meshO3d = meshO3d.simplify_quadric_decimation(target_number_of_triangles=targetTriangles)
triMesh = trimesh.Trimesh(vertices=np.asarray(meshO3d.vertices), faces=np.asarray(meshO3d.triangles))
numSurfacePoints = 10000
print(f"sampling {numSurfacePoints} points on mesh surface...")
surfacePc = meshO3d.sample_points_uniformly(number_of_points=numSurfacePoints)
surfacePointsNp = np.asarray(surfacePc.points)
dtype = torch.float16 if useHalf else torch.float32
surfacePoints = torch.tensor(surfacePointsNp, device=device, dtype=dtype)
bbox = meshO3d.get_axis_aligned_bounding_box()
globalBounds = [(bbox.min_bound[i], bbox.max_bound[i]) for i in range(3)]
verticesNp = np.asarray(meshO3d.vertices)
facesNp = np.asarray(meshO3d.triangles)
verticesTorch = torch.tensor(verticesNp, device=device, dtype=torch.float32)
facesTorch = torch.tensor(facesNp.astype(np.int64), device=device)
meshGpu = (verticesTorch, facesTorch)
minRadiusThreshold = 1e-8
spheresList = []
spheresTensor = None
if os.path.exists(jsonFilename):
    try:
        with open(jsonFilename, "r") as f:
            sphereData = json.load(f)
        spheresList = [(np.array(d["center"]), d["radius"]) for d in sphereData if d["radius"] > minRadiusThreshold]
        print(f"loaded {len(spheresList)} spheres from {jsonFilename}")
        if spheresList:
            spheresArray = np.array([np.hstack((center, [radius])) for center, radius in spheresList])
            spheresTensor = torch.tensor(spheresArray, device=device, dtype=dtype)
            totalSphereVolume = sum((4.0/3.0) * np.pi * (r ** 3) for _, r in spheresList)
            objVolume = triMesh.volume
            if objVolume > 0:
                fillFraction = totalSphereVolume / objVolume
                print(f"existing sphere packing fills {fillFraction*100:.2f}% of the object volume")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"error loading {jsonFilename}: {e}. starting from scratch.")
        spheresList = []
        spheresTensor = None
else:
    print(f"no existing {jsonFilename} found. starting from scratch.")
maxTotalSpheres = 333333333333
visualizer = o3d.visualization.Visualizer()
visualizer.create_window(window_name="Sphere Packing", width=800, height=600)
if spheresList:
    radii = [radius for _, radius in spheresList if radius > 0]
    minR = min(radii) if radii else 1
    maxR = max(radii) if radii else 1
    for center, radius in spheresList:
        color = getColorFromRadius(radius, minR, maxR)
        sphereMesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
        sphereMesh.translate(center)
        sphereMesh.compute_vertex_normals()
        sphereMesh.paint_uniform_color(color)
        visualizer.add_geometry(sphereMesh)
visualizer.poll_events()
visualizer.update_renderer()
print("\nstarting sphere packing (live display enabled)...")
sphereCount = len(spheresList)
while sphereCount < maxTotalSpheres:
    iterStartTime = time.time()
    numDirections = max(100, 1000 - sphereCount // 10)
    candidateList = findTangentSpheresBatch(
        spheresTensor,
        surfacePoints,
        meshGpu,
        globalBounds,
        numCandidates=numDirections,
        minRadius=minRadiusThreshold
    )
    if not candidateList:
        print("no candidate found, trying again with more directions.")
        numDirections *= 2
        candidateList = findTangentSpheresBatch(
            spheresTensor,
            surfacePoints,
            meshGpu,
            globalBounds,
            numCandidates=numDirections,
            minRadius=minRadiusThreshold
        )
        if not candidateList:
            print("still no candidates found. exiting.")
            break
    candCenter, candRadius = candidateList[0]
    if candRadius < minRadiusThreshold:
        print("candidate sphere too small, trying again.")
        continue
    print(f"sphere {sphereCount + 1}: center = {candCenter}, radius = {candRadius:.6f}")
    spheresList.append((candCenter, candRadius))
    newSphere = torch.tensor(np.hstack((candCenter, [candRadius])), device=device, dtype=dtype).unsqueeze(0)
    spheresTensor = newSphere if spheresTensor is None else torch.cat([spheresTensor, newSphere], dim=0)
    sphereCount += 1
    if sphereCount % 69 == 0:
        totalSphereVolume = sum((4.0/3.0) * np.pi * (r ** 3) for _, r in spheresList)
        objVolume = triMesh.volume
        if objVolume > 0:
            fillFraction = totalSphereVolume / objVolume
            print(f"volume filled: {fillFraction*100:.2f}% of the object")
    radii = [r for _, r in spheresList if r > 0]
    minR = min(radii) if radii else 1
    maxR = max(radii) if radii else 1
    color = getColorFromRadius(candRadius, minR, maxR)
    sphereMesh = o3d.geometry.TriangleMesh.create_sphere(radius=candRadius, resolution=12)
    sphereMesh.translate(candCenter)
    sphereMesh.compute_vertex_normals()
    sphereMesh.paint_uniform_color(color)
    visualizer.add_geometry(sphereMesh)
    visualizer.poll_events()
    visualizer.update_renderer()
    if sphereCount % 10 == 0:
        with open(jsonFilename, "w") as f:
            json.dump([{"center": center.tolist(), "radius": float(radius)}
                       for center, radius in spheresList], f, indent=4)
    iterEndTime = time.time()
    print(f"iteration time: {iterEndTime - iterStartTime:.4f} seconds")
    time.sleep(0.01)
visualizer.poll_events()
visualizer.update_renderer()
with open(jsonFilename, "w") as f:
    json.dump([{"center": center.tolist(), "radius": float(radius)}
               for center, radius in spheresList], f, indent=4)
print(f"\ntotal spheres packed: {sphereCount}")
print(f"total computation time: {time.time() - overallStartTime:.2f} seconds")
print("sphere packing complete. close the window to exit.")
while visualizer.poll_events():
    visualizer.update_renderer()
    time.sleep(0.01)
