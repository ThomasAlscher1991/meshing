# meshing
This is a small repository developed for object meshing in image registration.
It provides a rudimentary meshing functionality in 2D and 3D.

# 2D meshing
Provides 2D triangle meshes based on Delaunay triangulation. Convex shapes can be cleaned to avoid edges crossing over convex cavities.

# 3D meshing
Provides meshing for 3D triangle surface meshes and volumetric tetrahadral meshes.

# Installation
Installation via conda:

```git clone https://github.com/ThomasAlscher1991/meshing.git```

Navigate to directory to create conda environment and install requirements:

```conda env create --name MESHING --file requirements.yml```

Navigate to src and install meshing:

```python setup.py install```




