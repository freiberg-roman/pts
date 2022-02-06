import os
import meshio


def convert_to_stl():
    obj_dir = [x[0] for x in os.walk("./random_urdfs")][1:]

    d = obj_dir[0]
    file_base = d.split("/")[-1]
    file_base = "./random_urdfs/" + file_base + "/" + file_base

    # ### CONVERT ###

    mesh = meshio.read(file_base + ".obj")
    mesh_coll = meshio.read(file_base + "_coll.obj")
    mesh.write(file_base + ".stl")
    mesh_coll.write(file_base + "_coll.stl")


if __name__ == "__main__":
    convert_to_stl()
