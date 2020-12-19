function affiche_Reconstruction(x_r,x,Er,K)
figure();colormap gray;
subplot(1,2,1);imagesc(reshape(x,[64,64]));title('depart');
subplot(1,2,2);imagesc(reshape(x_r,[64,64]));title([num2str(K),'     ',num2str(Er)]);