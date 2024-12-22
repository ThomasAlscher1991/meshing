import numpy as np
import meshing
import os

tempDir=os.path.dirname(os.path.abspath(__file__))
mask=np.zeros((100,100,100))
mask[20:80,20:80,20:80]=1
mask[20:60,20:80,40:42]=0

surface3d=meshing.surfaceMesh( mask, tempDir, faceNumbers=1000,level=0.5)
verts,simps=surface3d.triangulate()

volume3d=meshing.tetrahedralMesh()
verts,simps=volume3d.getTetrahedralMeshFromMask(mask,tempDir, faceNumbers=1000,level=0.5)

