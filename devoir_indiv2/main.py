"""Side-by-side viewer: original mesh (left) vs simplified mesh (right).

Usage examples:
    python main.py                                   # obj3, edge collapse keeping 10 %
    python main.py DVI2/obj1.obj -m edge -k 0.2      # keep 20 % of the faces
    python main.py DVI2/obj6.obj -m cluster -r 60    # vertex clustering, 60 cells
    python main.py DVI2/obj2.obj --save              # also write the result to results/

Controls:
    mouse drag   rotate both models          scroll   zoom
    W            toggle wireframe            R        reset view
    Esc          quit
"""
import argparse
import time
from pathlib import Path

import glfw
import numpy as np
import OpenGL.GL as gl
import pyrr
from OpenGL.GL import shaders

from edge_collapse import edge_collapse
from obj_io import mesh_stats, read_obj, write_obj
from vertex_clustering import vertex_clustering

HERE = Path(__file__).resolve().parent

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 v_normal;
void main() {
    v_normal = mat3(model) * normal;
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
uniform vec3 color;
uniform bool lit;
out vec4 fragColor;
void main() {
    if (!lit) { fragColor = vec4(color, 1.0); return; }
    vec3 light = normalize(vec3(0.4, 0.6, 1.0));
    float diffuse = abs(dot(normalize(v_normal), light));   // two-sided
    fragColor = vec4(color * (0.25 + 0.75 * diffuse), 1.0);
}
"""


def mesh_arrays(obj_data):
    """Vertex positions, smooth vertex normals and triangle indices as numpy arrays."""
    vertices = np.asarray(obj_data["vertices"], dtype=np.float32)
    indices = np.asarray([[c[0] for c in face] for face in obj_data["faces"]], dtype=np.uint32)

    tri = vertices[indices]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, indices[:, k], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.where(lengths == 0, 1, lengths)
    return vertices, normals.astype(np.float32), indices.ravel()


def normalisation(vertices):
    """Centre/scale so the model fits in the unit sphere (used for both meshes)."""
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    scale = np.linalg.norm(vertices - center, axis=1).max()
    return center, (1.0 / scale if scale > 0 else 1.0)


class MeshComparer:
    def __init__(self, width=1280, height=720, title="Mesh Comparison"):
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)  # required on macOS

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create an OpenGL 3.3 window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.reset_view()
        self.wireframe = False
        self.is_dragging = False
        self.last_x = self.last_y = 0.0

        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_pos_callback)
        glfw.set_key_callback(self.window, self.key_callback)

        self.shader = shaders.compileProgram(
            shaders.compileShader(VERTEX_SHADER, gl.GL_VERTEX_SHADER),
            shaders.compileShader(FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER),
        )
        self.uniforms = {name: gl.glGetUniformLocation(self.shader, name)
                         for name in ("model", "view", "projection", "color", "lit")}
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.1, 0.1, 0.12, 1.0)

    # ------------------------------------------------------------ callbacks
    def reset_view(self):
        self.zoom, self.yaw, self.pitch = 3.0, 0.0, 0.0

    def scroll_callback(self, window, xoffset, yoffset):
        self.zoom = max(0.5, min(20.0, self.zoom - yoffset * 0.2))

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.is_dragging = action == glfw.PRESS
            self.last_x, self.last_y = glfw.get_cursor_pos(window)

    def mouse_pos_callback(self, window, x, y):
        if self.is_dragging:
            self.yaw += (x - self.last_x) * 0.01
            self.pitch = max(-1.5, min(1.5, self.pitch + (y - self.last_y) * 0.01))
            self.last_x, self.last_y = x, y

    def key_callback(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_W:
            self.wireframe = not self.wireframe
        elif key == glfw.KEY_R:
            self.reset_view()

    # --------------------------------------------------------------- meshes
    def load_mesh(self, vertices, normals, indices):
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)

        for location, data in ((0, vertices), (1, normals)):
            vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
            gl.glEnableVertexAttribArray(location)
            gl.glVertexAttribPointer(location, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)

        gl.glBindVertexArray(0)
        return {"vao": vao, "count": len(indices)}

    def draw(self, mesh, color):
        gl.glUniform3f(self.uniforms["color"], *color)
        gl.glBindVertexArray(mesh["vao"])
        if self.wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            gl.glUniform1i(self.uniforms["lit"], 0)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glUniform1i(self.uniforms["lit"], 1)
        gl.glDrawElements(gl.GL_TRIANGLES, mesh["count"], gl.GL_UNSIGNED_INT, None)

    def compare_meshes(self, original_data, simplified_data):
        original = mesh_arrays(original_data)
        simplified = mesh_arrays(simplified_data)

        # Same transform for both meshes so they stay comparable
        center, scale = normalisation(original[0])
        for arrays in (original, simplified):
            arrays[0][:] = (arrays[0] - center) * scale

        original_mesh = self.load_mesh(*original)
        simplified_mesh = self.load_mesh(*simplified)

        try:
            while not glfw.window_should_close(self.window):
                fb_width, fb_height = glfw.get_framebuffer_size(self.window)
                half = max(fb_width // 2, 1)
                gl.glViewport(0, 0, fb_width, fb_height)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

                view = pyrr.matrix44.create_look_at((0, 0, self.zoom), (0, 0, 0), (0, 1, 0))
                projection = pyrr.matrix44.create_perspective_projection(
                    45, half / max(fb_height, 1), 0.01, 100.0)
                model = pyrr.matrix44.multiply(
                    pyrr.matrix44.create_from_y_rotation(self.yaw),
                    pyrr.matrix44.create_from_x_rotation(self.pitch),
                )

                gl.glUseProgram(self.shader)
                gl.glUniformMatrix4fv(self.uniforms["view"], 1, gl.GL_FALSE, view.astype(np.float32))
                gl.glUniformMatrix4fv(self.uniforms["projection"], 1, gl.GL_FALSE,
                                      projection.astype(np.float32))
                gl.glUniformMatrix4fv(self.uniforms["model"], 1, gl.GL_FALSE, model.astype(np.float32))

                gl.glViewport(0, 0, half, fb_height)
                self.draw(original_mesh, (0.2, 0.5, 0.8))
                gl.glViewport(half, 0, fb_width - half, fb_height)
                self.draw(simplified_mesh, (0.85, 0.45, 0.2))

                glfw.swap_buffers(self.window)
                glfw.poll_events()
        finally:
            glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=HERE / "DVI2" / "obj3.obj")
    parser.add_argument("-m", "--method", choices=("edge", "cluster"), default="edge")
    parser.add_argument("-k", "--keep", type=float, default=0.1,
                        help="[edge] fraction of faces to keep (default 0.1)")
    parser.add_argument("--cost", choices=("quadric", "length"), default="quadric",
                        help="[edge] collapse cost (default quadric)")
    parser.add_argument("-r", "--resolution", type=int, default=30,
                        help="[cluster] grid cells along the longest side (default 30)")
    parser.add_argument("--save", action="store_true", help="write the simplified mesh to results/")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    original_data = read_obj(args.input)

    t0 = time.perf_counter()
    if args.method == "edge":
        simplified_data = edge_collapse(original_data, args.keep, cost=args.cost)
        label = f"edge collapse ({args.cost}, keep {args.keep:.0%})"
    else:
        simplified_data = vertex_clustering(original_data, resolution=args.resolution)
        label = f"vertex clustering ({args.resolution} cells)"
    elapsed = time.perf_counter() - t0

    (v0, f0), (v1, f1) = mesh_stats(original_data), mesh_stats(simplified_data)
    print(f"Original mesh   : {v0} vertices, {f0} faces")
    print(f"Simplified mesh : {v1} vertices, {f1} faces")
    print(f"Faces kept      : {f1 / f0:.2%}  ({label}, {elapsed:.2f}s)")

    if args.save:
        out = HERE / "results" / f"{Path(args.input).stem}_{args.method}.obj"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_obj(simplified_data, out)
        print(f"Saved {out}")

    title = f"Original {f0} faces  |  {label}: {f1} faces"
    MeshComparer(title=title).compare_meshes(original_data, simplified_data)


if __name__ == "__main__":
    main()
