function Er = erreur_Reconstruction(x_r,x)
Er = sqrt(sum((x - x_r).^2));
end

