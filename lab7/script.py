import OpenGL.GL as gl
import glfw
import OpenGL.GLU as glu
import numpy as np
import math
import OpenGL.GL as gl

# Dimensions de la grille pour la surface
GRID_SIZE = 20

# Points de contrôle (initialisés avec une grille régulière)
control_points = np.array([
    [[-1.0, -1.0, 0.0], [-1.0, 0.0, 0.5], [-1.0, 1.0, 0.0]],
    [[ 0.0, -1.0, 0.5], [ 0.0, 0.0, 1.0], [ 0.0, 1.0, 0.5]],
    [[ 1.0, -1.0, 0.0], [ 1.0, 0.0, 0.5], [ 1.0, 1.0, 0.0]]
], dtype=np.float32)

# Paramètres de contrôle
current_point = [0, 0]  # Indice du point de contrôle courant
signe = 1               # Direction de modification (+1 ou -1)
pas = 0.1              # Pas de modification


def bernstein_poly(i, n, t):
    """Calcul d'un polynôme de Bernstein."""
    return math.comb(n, i) * (t**i) * ((1 - t)**(n - i))


def bezier_surface(u, v):
    """Calcul d'un point sur la surface de Bézier."""
    n = control_points.shape[0] - 1
    m = control_points.shape[1] - 1
    point = np.zeros(3)
    for i in range(n + 1):
        for j in range(m + 1):
            b = bernstein_poly(i, n, u) * bernstein_poly(j, m, v)
            point += b * control_points[i][j]
    return point


def draw_surface():
    """Dessiner la surface de Bézier avec des normales."""
    gl.glBegin(gl.GL_QUADS)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            u1, v1 = i / GRID_SIZE, j / GRID_SIZE
            u2, v2 = (i + 1) / GRID_SIZE, j / GRID_SIZE
            u3, v3 = (i + 1) / GRID_SIZE, (j + 1) / GRID_SIZE
            u4, v4 = i / GRID_SIZE, (j + 1) / GRID_SIZE

            p1 = bezier_surface(u1, v1)
            p2 = bezier_surface(u2, v2)
            p3 = bezier_surface(u3, v3)
            p4 = bezier_surface(u4, v4)

            # Calculer une normale approximative pour la face
            normal = np.cross(p2 - p1, p3 - p1)
            normal = normal / np.linalg.norm(normal)

            gl.glNormal3fv(normal)  # Appliquer la normale
            gl.glVertex3fv(p1)
            gl.glVertex3fv(p2)
            gl.glVertex3fv(p3)
            gl.glVertex3fv(p4)
    gl.glEnd()


def draw_control_points():
    """Dessiner les points de contrôle."""
    gl.glPointSize(10.0)
    gl.glBegin(gl.GL_POINTS)
    for i in range(control_points.shape[0]):
        for j in range(control_points.shape[1]):
            if [i, j] == current_point:
                gl.glColor3f(1.0, 0.0, 0.0)  # Point courant en rouge
            else:
                gl.glColor3f(0.0, 1.0, 0.0)  # Autres points en vert
            gl.glVertex3fv(control_points[i][j])
    gl.glEnd()


def key_callback(window, key, scancode, action, mods):
    """Gestion des entrées clavier."""
    global signe, current_point

    if action != glfw.PRESS:
        return

    if key == glfw.KEY_X:
        control_points[current_point[0]][current_point[1]][0] += signe * pas
    elif key == glfw.KEY_Y:
        control_points[current_point[0]][current_point[1]][1] += signe * pas
    elif key == glfw.KEY_Z:
        control_points[current_point[0]][current_point[1]][2] += signe * pas
    elif key == glfw.KEY_S:
        signe *= -1
    elif key == glfw.KEY_TAB:
        current_point[1] += 1
        if current_point[1] >= control_points.shape[1]:
            current_point[1] = 0
            current_point[0] = (current_point[0] + 1) % control_points.shape[0]


def main():
    """Programme principal."""
    if not glfw.init():
        return

    # Création de la fenêtre
    window = glfw.create_window(800, 600, "Surface de Bézier 3D", None, None)
    if not window:
        glfw.terminate()
        return

    # Définir le contexte OpenGL courant
    glfw.make_context_current(window)

    # Enregistrer le callback pour le clavier
    glfw.set_key_callback(window, key_callback)

    # Initialisation OpenGL
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Configuration de la projection
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(45, 800/600, 1.0, 10.0)
    gl.glMatrixMode(gl.GL_MODELVIEW)

    # Boucle principale
    while not glfw.window_should_close(window):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # Position de la caméra
        glu.gluLookAt(0.0, 0.0, 5.0,  # Position de la caméra
                      0.0, 0.0, 0.0,  # Point à regarder
                      0.0, 1.0, 0.0)  # Orientation (axe Y vers le haut)

        # Dessiner la surface
        gl.glColor3f(0.5, 0.5, 1.0)
        draw_surface()

        # Dessiner les points de contrôle
        draw_control_points()

        # Échange des buffers et traitement des événements
        glfw.swap_buffers(window)
        glfw.poll_events()

    # Nettoyage
    glfw.terminate()


if __name__ == "__main__":
    main()