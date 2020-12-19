function J = sobel_apres_lissage(Il,seuil)

 [Ix,Iy] = sobel(Il);
 M = module(Ix,Iy);
 J = detecteur(M,seuil);
