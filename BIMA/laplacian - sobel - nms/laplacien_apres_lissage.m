function J = laplacien_apres_lissage(Il,seuil)
 J1 = laplacien(Il);
 J = passage(J1,seuil);