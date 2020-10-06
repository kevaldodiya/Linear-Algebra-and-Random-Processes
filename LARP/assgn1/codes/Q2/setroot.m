function[sigma] = setroot(D,n,m)
if n == m
   
    sigma = zeros(size(D,1));     
    for i= 1:size(D,1)
        sigma(i,i) =sqrt(D(i,i));
    end
elseif n<m
    sigma = zeros(n,m);
    for i=1:n
        sigma(i,i) = sqrt(D(i,i));
    end
else
    sigma = zeros(n,m);
    for i=1:m
        sigma(i,i) = sqrt(D(i,i));
    end
end
end
        