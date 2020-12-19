function [err,err_mean] = angle(wr,we,pad)
err = acos( (1+sum(wr.*we,3)) ./ sqrt( (1+sum(wr.^2,3)).*(1+sum(we.^2,3)) )  );
err_mean = mean(err(1+pad:end-pad,1+pad:end-pad),'all');
end