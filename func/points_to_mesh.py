"""Module to generate .obj file from bronchus coords outer shell
Needs stage-02 (reduced_model.npz) as input.
This file expects the first argument to be a path to the input folder (not including the 
reduced_model.npz). The second argument needs to be a path to the output file.
It does so by going through every bronchus point in the reduced_model.npz and checking
each of the 6 neighboring points whether it is empty. If it is indeed empty then
it adds the points which haven't been added yet on that face and also adds the face.
Then it saves everything into a .obj file with the format of [patient_it].obj. This file can
be imported into blender, 3D printed, visualized and many other nice things.
"""
from collections import defaultdict
from pathlib import Path
from typing import List, Set, Tuple, Dict
import numpy as np
from skimage.morphology import skeletonize


def generate_obj(
    output_data_path: Path,
    accepted_types: Set[int],
    model: np.ndarray,
    color_mask: np.ndarray = None,
    color_to_rgb_tuple: Dict[int, Tuple[float, float, float]] = {},
    rot_mat: np.ndarray = None,
    num_decimal_digits: int = 2,
):
    """Saves a .obj obj_file given the model, the accepted types and a name
    output_data_path is a pathlib Path, this is the full path the obj_file will be saved as
    accepted_types is a list or set which types should be looked for, other
    types will be ignored. If empty set then everything except for 0 will be
    accepted
    model is the 3D numpy array model of the lung or whatever object you
    want to convert to 3D
    color_to_rgb_tuple is a Dict which maps the color id used in color mask to a tuple
    of rgb values. This color will be used to color the object with that color.
    (e.g. {1: (1, 0.4, 0.5)})
    color_mask is a model with the same shape as model, but its numbers represent
    groups of colors/materials which will be added by this script
    rot_mat is a rotation matrix. Each point p=(x,y,z) is rotated by rot_mat @ p or left unchanged if None
    """

    occurrences = np.unique(model)
    if not all(t in occurrences for t in accepted_types):
        return

    output_data_path = Path(output_data_path)

    print(f"Generating {output_data_path} with accepted types of {accepted_types}")

    vertices = {}
    faces: Dict[int, List[List[int]]] = defaultdict(list)

    model = np.pad(np.copy(model), 1)
    if color_mask is not None:
        color_mask = np.pad(color_mask, 1)
    if accepted_types:
        for remove in set(np.unique(model)) - accepted_types - {0}:
            model[model == remove] = 0

    index = 1
    # Iterate over each axis and pos/neg directions, then roll the model over, afterwards subtracting these.
    # This causes there to be only -1 and 1 values where there is air, meaning there a face should be added.
    # Though we only look at the 1 values, since these are actually in the model (the others are outside, which means
    # their color map will be wrong)
    for axis in range(3):
        for pos_or_neg in [-1, 1]:
            diff = np.roll(model, -pos_or_neg, axis=axis)
            model_diff = np.where(model > diff)
            coords = []
            for d1, d2 in [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]:
                coords.append(list(map(lambda t: t.astype(float), np.copy(model_diff))))
                coords[-1][axis] += pos_or_neg / 2
                coords[-1][(axis + 1) % 3] += d1
                coords[-1][(axis + 2) % 3] += d2

            # Example shape after transpose: (11523, 4, 3) - face_coords is an array of faces,
            # each face is a list of 4 points with 3 coordinates.
            faces_coords = np.transpose(coords, axes=(2, 0, 1))
            vertex_coords = np.transpose(model_diff)
            for vertex_coord, face_coords in zip(vertex_coords, faces_coords):
                curr_face = []
                for face_coord in map(tuple, face_coords):
                    if face_coord not in vertices:
                        vertices[face_coord] = index
                        index += 1
                    curr_face.append(vertices[face_coord])
                material = color_mask[tuple(vertex_coord)] if color_mask is not None else 0
                faces[material].append(curr_face)
                # assert len(faces[material][-1]) == 4, f"ERROR: Wrong number of points on face {faces[material][-1]}"

    print(f"Vertex count : {len(vertices):,}")
    print(f"Face count : {sum(map(len, faces.values())):,}")

    # make to numpy for easier usage later
    vertices = np.array([np.array(v) for v in vertices])
    vertices = normalize(vertices, model.shape, rot_mat=rot_mat)

    # Write vertices and faces to obj_file
    material_path = output_data_path.with_suffix(".mtl")
    with open(material_path, "w") as mat_file:
        import random

        random.seed(output_data_path.parent.name)

        def ran():
            return random.uniform(0, 1)

        for material in faces:
            mat_file.write(f"newmtl mat{material}\n")
            mat_file.write("Ns 96.078431\n")
            mat_file.write("Ka 1.000000 1.000000 1.000000\n")
            rgb = color_to_rgb_tuple[material] if material in color_to_rgb_tuple else (ran(), ran(), ran())
            mat_file.write(f"Kd {' '.join(map(str, rgb))}\n")
            mat_file.write("Ks 0.500000 0.500000 0.500000\n")
            mat_file.write("Ke 0.000000 0.000000 0.000000\n")
            mat_file.write("Ni 1.000000\n")
            mat_file.write("d 1.000000\n")
            mat_file.write("illum 2\n\n")
    with open(output_data_path, "w") as obj_file:
        obj_file.write("# .obj generated by Airway")
        obj_file.write(f"mtllib {material_path.name}\n")
        obj_file.write("# Vertices\n")
        # original was [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
        vertexformat = f"v {{:.{num_decimal_digits}f}} {{:.{num_decimal_digits}f}} {{:.{num_decimal_digits}f}}\n"
        for x, y, z in vertices:
            obj_file.write(vertexformat.format(x, y, z))

        obj_file.write("\n# Faces\n")
        for material, faces_with_material in faces.items():
            obj_file.write(f"usemtl mat{material}\n")
            for a, b, c, d in faces_with_material:
                obj_file.write(f"f {a} {b} {c} {d}\n")


def normalize(vertices: np.ndarray, reference_shape: np.ndarray, rot_mat: np.ndarray = None):
    # Shift to middle of the space
    vertices -= np.array(reference_shape) / 2
    # Scale to [-10..10]
    vertices *= 20 / np.max(reference_shape)
    # If available: transform
    # Note: since this is applied afterwards, points can be out of [-10..10]
    if rot_mat is not None:
        vertices = vertices @ np.transpose(rot_mat)
    return vertices


def main():

    model = np.load("model.npz")["arr_0"]
    print(f"Loaded model with shape {model.shape}")

    bronchus_color_mask = None
    color_codes = {0: (1, 1, 1)}

    rot_mat = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    print("Running skeletonize on model")
    # Remove lobe coordinates from model by clipping everything
    # between 0 and 2, then modulo everything by 2 to remove 2s
    skeleton = skeletonize(np.clip(model, 0, 2) % 2)
    generate_obj("skeleton.obj", set(), skeleton, rot_mat=rot_mat)
    generate_obj(
        "3dobj.obj",
        {1},
        model,
        color_mask=bronchus_color_mask,
        color_to_rgb_tuple=color_codes,
        rot_mat=rot_mat,
    )

def produce_3d_obj(input_img, output_file_name):
    
    model = input_img
    print(f"Loaded model with shape {model.shape}")

    bronchus_color_mask = None
    color_codes = {0: (1, 1, 1)}

    rot_mat = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    print("Running skeletonize on model")
    # Remove lobe coordinates from model by clipping everything
    # between 0 and 2, then modulo everything by 2 to remove 2s
    skeleton = skeletonize(np.clip(model, 0, 2) % 2)
    generate_obj(output_file_name+"_skeleton.obj", set(), skeleton, rot_mat=rot_mat)
    generate_obj(
        output_file_name+"_3dobj.obj",
        {1},
        model,
        color_mask=bronchus_color_mask,
        color_to_rgb_tuple=color_codes,
        rot_mat=rot_mat,
    )

if __name__ == "__main__":
    main()