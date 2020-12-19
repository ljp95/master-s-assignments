function [Ir,ordom] = rectify_orientation(I)
M = to_visualize_TF(compute_FT(I));
Mb = seuillerImage(M,3*10^5);
[Ior,ordom] = orientationDominante(Mb);
Ir=rotationImage(I,-ordom);
