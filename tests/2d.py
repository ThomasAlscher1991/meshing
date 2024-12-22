import numpy as np
import matplotlib.pyplot as plt
import meshing
import matplotlib.tri as mtri

mask=np.zeros((100,100))
mask[20:80,20:80]=1
mask[20:60,40:41]=0

mesher2d=meshing.triangleMesh()
mesher2d.getTriangleMeshFromMask(mask=mask)
verts,simps=mesher2d.getVerticesAndSimplices()
mesher2d.cleanConvex(simps,verts,mask)
verts,simps=mesher2d.getVerticesAndSimplices()


###Visual
"""triangulation = mtri.Triangulation(verts[:,1], verts[:,0], simps)
plt.figure(figsize=(6, 6))
plt.imshow(mask)
plt.triplot(triangulation, color='blue')  # Wireframe
plt.scatter(verts[:,1], verts[:,0], color='red')  # Vertices
plt.title("Triangular Mesh")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().set_aspect('equal')
plt.show()"""
