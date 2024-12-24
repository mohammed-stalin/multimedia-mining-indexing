import glfw
import OpenGL.GL as gl
import numpy as np
from OpenGL.GL import shaders
import pyrr

class MeshComparer:
    def __init__(self, width=1280, height=720):
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        
        self.window = glfw.create_window(width, height, "Mesh Comparison", None, None)
        glfw.make_context_current(self.window)
        self.zoom = 3.0  # Initial camera distance
    
        # Add scroll callback
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        
        self.is_dragging = False
        self.last_x = 0
        self.last_y = 0
        self.yaw = 0.0
        self.pitch = 0.0
        
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_pos_callback)
        
        vertex_shader = """
        #version 330
        layout(location = 0) in vec3 position;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * model * vec4(position, 1.0);
        }
        """
        
        fragment_shader = """
        #version 330
        uniform vec3 color;
        out vec4 fragColor;
        void main() {
            fragColor = vec4(color, 1.0);
        }
        """
        
        self.shader = shaders.compileProgram(
            shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
            shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER)
        )
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        self.view = pyrr.matrix44.create_look_at(
            (0, 0, 3), (0, 0, 0), (0, 1, 0)
        )
        self.projection = pyrr.matrix44.create_perspective_projection(
            45, width/height, 0.1, 100.0
        )
    def scroll_callback(self, window, xoffset, yoffset):
        self.zoom -= yoffset * 0.1  # Adjust zoom speed
        self.zoom = max(0.1, min(20.0, self.zoom))  # Limit zoom range

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.is_dragging = True
                x, y = glfw.get_cursor_pos(window)
                self.last_x = x
                self.last_y = y
            elif action == glfw.RELEASE:
                self.is_dragging = False

    def mouse_pos_callback(self, window, x, y):
        if self.is_dragging:
            dx = x - self.last_x
            dy = y - self.last_y
            
            self.yaw += dx * 0.005
            self.pitch += dy * 0.005
            self.pitch = max(min(self.pitch, 1.5), -1.5)
            
            self.last_x = x
            self.last_y = y

    def load_mesh(self, vertices, faces):
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        
        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, gl.GL_STATIC_DRAW)
        
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        
        return {'vao': vao, 'num_faces': len(faces)}

    def process_faces(self, faces):
        # Extract vertex indices from face tuples and create triangles
        triangles = []
        for face in faces:
            # Get vertex indices from face tuples
            vertex_indices = [vertex_tuple[0] - 1 for vertex_tuple in face]  # Subtract 1 for 0-based indexing
            triangles.extend(vertex_indices)
        return np.array(triangles, dtype=np.uint32)

    def compare_meshes(self, original_data, simplified_data):
        try:
            # Convert vertices to numpy arrays
            original_vertices = np.array(original_data['vertices'], dtype=np.float32)
            simplified_vertices = np.array(simplified_data['vertices'], dtype=np.float32)
            
            # Process faces
            original_faces = self.process_faces(original_data['faces'])
            simplified_faces = self.process_faces(simplified_data['faces'])
            
            self.original_mesh = self.load_mesh(original_vertices, original_faces)
            self.simplified_mesh = self.load_mesh(simplified_vertices, simplified_faces)
            
            print(f"Original mesh: {len(original_vertices)} vertices, {len(original_data['faces'])} faces")
            print(f"Simplified mesh: {len(simplified_vertices)} vertices, {len(simplified_data['faces'])} faces")
            print(f"Reduction ratio: {len(simplified_data['faces'])/len(original_data['faces']):.2%}")
            
            while not glfw.window_should_close(self.window):
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                # Update view matrix with zoom
                self.view = pyrr.matrix44.create_look_at(
                    (0, 0, self.zoom),  # Camera position
                    (0, 0, 0),         # Target
                    (0, 1, 0)          # Up vector
                )
                rot_x = pyrr.Matrix44.from_x_rotation(self.pitch)
                rot_y = pyrr.Matrix44.from_y_rotation(self.yaw)
                model = pyrr.matrix44.multiply(rot_x, rot_y)
                
                gl.glUseProgram(self.shader)
                
                gl.glUniformMatrix4fv(
                    gl.glGetUniformLocation(self.shader, "view"),
                    1, gl.GL_FALSE, self.view
                )
                gl.glUniformMatrix4fv(
                    gl.glGetUniformLocation(self.shader, "projection"),
                    1, gl.GL_FALSE, self.projection
                )
                gl.glUniformMatrix4fv(
                    gl.glGetUniformLocation(self.shader, "model"),
                    1, gl.GL_FALSE, model
                )
                
                # Draw original mesh
                gl.glViewport(0, 0, 640, 720)
                gl.glUniform3f(gl.glGetUniformLocation(self.shader, "color"), 0.2, 0.5, 0.8)
                gl.glBindVertexArray(self.original_mesh['vao'])
                gl.glDrawElements(gl.GL_TRIANGLES, self.original_mesh['num_faces'], gl.GL_UNSIGNED_INT, None)
                
                # Draw simplified mesh
                gl.glViewport(640, 0, 640, 720)
                gl.glUniform3f(gl.glGetUniformLocation(self.shader, "color"), 0.8, 0.4, 0.2)
                gl.glBindVertexArray(self.simplified_mesh['vao'])
                gl.glDrawElements(gl.GL_TRIANGLES, self.simplified_mesh['num_faces'], gl.GL_UNSIGNED_INT, None)
                
                glfw.swap_buffers(self.window)
                glfw.poll_events()
            
        except Exception as e:
            print(f"Error processing meshes: {str(e)}")
        finally:
            glfw.terminate()

def main():
    from read_obj import read_obj
    from vertex_clustering import vertex_clustering
    from edge_collapse import edge_collapse
    
    original_data = read_obj('DVI2/obj6.obj')
    simplified_data = edge_collapse(original_data, ratio=0.1)
    
    comparer = MeshComparer()
    comparer.compare_meshes(original_data, simplified_data)

if __name__ == '__main__':
    main()