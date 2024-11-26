import numpy as np
import cv2 as cv
from random import randrange


drawing = False  
ix, iy = -1, -1  # Coordonnées du début du trait du marquer
current_marker = 1  # Marqueur courant (étiquette)
color = (0, 255, 0)  # Couleur initiale (vert)
thickness = 1  # Épaisseur du trait par défaut
markers = None  # Tableau pour stocker les marqueurs
colors = []  # Couleurs aléatoires pour chaque marqueur


img = cv.imread('C:/Users/lenovo/Desktop/multimedia-mining-indexing/lab6/coins.png')
original_img = img.copy()  # Copie pour réinitialisation
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

# Préparation des marqueurs (tableu de meme taille de l'image)
markers = np.zeros_like(gray, dtype=np.int32)

# Couleurs aléatoires pour chaque marqueur
colors = [[randrange(256), randrange(256), randrange(256)] for _ in range(10)]

# Fonction pour dessiner des contours interactifs
def draw_marker(event, x, y, flags, param):
    global img, drawing, ix, iy, current_marker, thickness

    #lorsque left button est cliqué
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    #mise a jour des coordonnes lorsque la sourie en mvt
    elif event == cv.EVENT_MOUSEMOVE and drawing:
        cv.line(img, (ix, iy), (x, y), color, thickness)
        cv.line(markers, (ix, iy), (x, y), current_marker, thickness)
        ix, iy = x, y

    #lorsque left button est laché
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.line(img, (ix, iy), (x, y), color, thickness)
        cv.line(markers, (ix, iy), (x, y), current_marker, thickness)


cv.namedWindow('Dessinez vos marqueurs')
cv.setMouseCallback('Dessinez vos marqueurs', draw_marker)

while True:
    cv.imshow('Dessinez vos marqueurs', img)
    k = cv.waitKey(1) & 0xFF

    if k == ord('q'):  # Quitter
        break
    elif k == ord('c'):  # Réinitialiser
        img = original_img.copy()
        markers = np.zeros_like(gray, dtype=np.int32)
    elif k == ord('n'):  # Changer de marqueur
        current_marker += 1
        color = (randrange(256), randrange(256), randrange(256))
    elif k == ord('+'):  # Augmenter l'épaisseur
        thickness += 1
    elif k == ord('-') and thickness > 1:  # Diminuer l'épaisseur
        thickness -= 1

cv.destroyWindow('Dessinez vos marqueurs')

# Appliquer l'algorithme Ligne de partage des eaux
markers[markers > 0] += 1  # Décaler les marqueurs pour éviter les conflits avec le fond
kernel = np.ones((3, 3), np.uint8)
fond = cv.dilate(thresh, kernel, iterations=12)
markers[fond == 0] = 1

# Algorithme Ligne de partage des eaux
markers_watershed = cv.watershed(original_img, markers)

# Générer l'image segmentée
result = np.zeros_like(original_img)
for label in range(2, markers_watershed.max() + 1):
    result[markers_watershed == label] = colors[label % len(colors)]

# Marquer les lignes de partage des eaux en rouge
result[markers_watershed == -1] = [255, 0, 0]

# Affichage du résultat
cv.imshow('Segmentation', result)
cv.waitKey(0)
cv.destroyAllWindows()
