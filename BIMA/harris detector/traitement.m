function R1 = traitement(I, echelle, seuil)
R = calculR(I,echelle);
Rb = seuilleR(R,seuil);
Rnms = nms(R,Rb);
R1 = affichePts(I,Rnms,echelle);
end