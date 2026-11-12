"""Lab 7 - Interactive 3D Bezier surface (OpenGL + GLFW).

A bi-quadratic (3x3 control points) Bezier surface

    S(u, v) = sum_i sum_j B_i^n(u) B_j^m(v) P_ij,   B_i^n(t) = C(n,i) t^i (1-t)^(n-i)

is evaluated on a GRID_SIZE x GRID_SIZE grid and rendered with lighting.
The control points can be edited from the keyboard.

Controls
    Tab          select the next control point (shown in red)
    X / Y / Z    move the selected point along that axis by +/- STEP
    S            flip the direction of movement (+ / -)
    W            toggle wireframe / filled surface
    N            toggle control net display
    R            reset control points and view
    mouse drag   rotate the scene        scroll  zoom
    Esc          quit
"""
import math

import glfw
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu

GRID_SIZE = 20
STEP = 0.1
WIDTH, HEIGHT = 900, 700

INITIAL_CONTROL_POINTS = np.array([
    [[-1.0, -1.0, 0.0], [-1.0, 0.0, 0.5], [-1.0, 1.0, 0.0]],
    [[0.0, -1.0, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 0.5]],
    [[1.0, -1.0, 0.0], [1.0, 0.0, 0.5], [1.0, 1.0, 0.0]],
], dtype=np.float64)


def bernstein_matrix(n, t):
    """B[k, i] = B_i^n(t_k) for every parameter value t_k."""
    t = np.asarray(t)[:, None]
    i = np.arange(n + 1)[None, :]
    coeffs = np.array([math.comb(n, k) for k in range(n + 1)])[None, :]
    return coeffs * t ** i * (1 - t) ** (n - i)


def bezier_surface(control_points, resolution=GRID_SIZE):
    """Evaluate the surface on a (resolution+1)^2 grid -> array (R+1, R+1, 3)."""
    n, m = control_points.shape[0] - 1, control_points.shape[1] - 1
    t = np.linspace(0.0, 1.0, resolution + 1)
    bu, bv = bernstein_matrix(n, t), bernstein_matrix(m, t)
    return np.einsum("ui,vj,ijk->uvk", bu, bv, control_points)


def vertex_normals(points):
    """Smooth per-vertex normals from the grid's finite differences."""
    du = np.gradient(points, axis=0)
    dv = np.gradient(points, axis=1)
    normals = np.cross(du, dv)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.where(norm == 0, 1, norm)


class BezierApp:
    def __init__(self):
        self.reset()
        self.wireframe = False
        self.show_net = True
        self.dragging = False
        self.last_cursor = (0.0, 0.0)

    def reset(self):
        self.control_points = INITIAL_CONTROL_POINTS.copy()
        self.current = [0, 0]
        self.sign = 1
        self.yaw, self.pitch, self.distance = 30.0, -60.0, 5.0
        self.dirty = True

    # ----------------------------------------------------------- callbacks
    def on_key(self, window, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        i, j = self.current
        axis = {glfw.KEY_X: 0, glfw.KEY_Y: 1, glfw.KEY_Z: 2}.get(key)
        if axis is not None:
            self.control_points[i, j, axis] += self.sign * STEP
            self.dirty = True
        elif key == glfw.KEY_S:
            self.sign *= -1
        elif key == glfw.KEY_TAB:
            j += 1
            if j >= self.control_points.shape[1]:
                j, i = 0, (i + 1) % self.control_points.shape[0]
            self.current = [i, j]
        elif key == glfw.KEY_W:
            self.wireframe = not self.wireframe
        elif key == glfw.KEY_N:
            self.show_net = not self.show_net
        elif key == glfw.KEY_R:
            self.reset()
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        self.update_title(window)

    def on_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == glfw.PRESS
            self.last_cursor = glfw.get_cursor_pos(window)

    def on_cursor(self, window, x, y):
        if self.dragging:
            dx, dy = x - self.last_cursor[0], y - self.last_cursor[1]
            self.yaw += dx * 0.4
            self.pitch = max(-90.0, min(90.0, self.pitch + dy * 0.4))
            self.last_cursor = (x, y)

    def on_scroll(self, window, xoff, yoff):
        self.distance = max(2.0, min(15.0, self.distance - yoff * 0.3))

    def on_resize(self, window, width, height):
        if height == 0:
            return
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45, width / height, 0.1, 50.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def update_title(self, window):
        i, j = self.current
        p = self.control_points[i, j]
        glfw.set_window_title(
            window,
            f"Bezier surface | point ({i},{j}) = ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) "
            f"| direction {'+' if self.sign > 0 else '-'}",
        )

    # ------------------------------------------------------------- drawing
    def draw_surface(self):
        if self.dirty:
            self.points = bezier_surface(self.control_points)
            self.normals = vertex_normals(self.points)
            self.dirty = False
        pts, nrm = self.points, self.normals

        gl.glEnable(gl.GL_LIGHTING)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.wireframe else gl.GL_FILL)
        gl.glColor3f(0.45, 0.55, 1.0)
        for i in range(pts.shape[0] - 1):
            gl.glBegin(gl.GL_QUAD_STRIP)
            for j in range(pts.shape[1]):
                gl.glNormal3dv(nrm[i, j])
                gl.glVertex3dv(pts[i, j])
                gl.glNormal3dv(nrm[i + 1, j])
                gl.glVertex3dv(pts[i + 1, j])
            gl.glEnd()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDisable(gl.GL_LIGHTING)

    def draw_control_net(self):
        cp = self.control_points
        if self.show_net:
            gl.glColor3f(0.7, 0.7, 0.7)
            gl.glLineWidth(1.0)
            for row in list(cp) + list(cp.transpose(1, 0, 2)):
                gl.glBegin(gl.GL_LINE_STRIP)
                for p in row:
                    gl.glVertex3dv(p)
                gl.glEnd()

        gl.glDisable(gl.GL_DEPTH_TEST)  # keep control points visible through the surface
        gl.glPointSize(10.0)
        gl.glBegin(gl.GL_POINTS)
        for i in range(cp.shape[0]):
            for j in range(cp.shape[1]):
                if [i, j] == self.current:
                    gl.glColor3f(1.0, 0.0, 0.0)
                else:
                    gl.glColor3f(0.0, 1.0, 0.0)
                gl.glVertex3dv(cp[i, j])
        gl.glEnd()
        gl.glEnable(gl.GL_DEPTH_TEST)

    # ---------------------------------------------------------------- main
    def run(self):
        if not glfw.init():
            raise RuntimeError("GLFW initialisation failed")
        window = glfw.create_window(WIDTH, HEIGHT, "Bezier surface", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Could not create the GLFW window")
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        glfw.set_key_callback(window, self.on_key)
        glfw.set_mouse_button_callback(window, self.on_mouse_button)
        glfw.set_cursor_pos_callback(window, self.on_cursor)
        glfw.set_scroll_callback(window, self.on_scroll)
        glfw.set_framebuffer_size_callback(window, self.on_resize)

        gl.glClearColor(0.08, 0.08, 0.1, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glLightModeli(gl.GL_LIGHT_MODEL_TWO_SIDE, gl.GL_TRUE)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (2.0, 3.0, 5.0, 0.0))
        gl.glEnable(gl.GL_POINT_SMOOTH)

        self.on_resize(window, *glfw.get_framebuffer_size(window))
        self.update_title(window)
        print(__doc__)

        while not glfw.window_should_close(window):
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glLoadIdentity()
            gl.glTranslatef(0.0, 0.0, -self.distance)
            gl.glRotatef(self.pitch, 1.0, 0.0, 0.0)
            gl.glRotatef(self.yaw, 0.0, 0.0, 1.0)

            self.draw_surface()
            self.draw_control_net()

            glfw.swap_buffers(window)
            glfw.poll_events()

        glfw.terminate()


if __name__ == "__main__":
    BezierApp().run()
