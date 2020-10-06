A = imread('sq31.jpg');
A = rgb2gray(A);
A = im2double(A);
Atrans_A = transpose(A)*A;
[P2,D2] = eig(Atrans_A);
[P2,D2] = setEigen(P2,D2);
sigma = setroot(D2,size(A,1),size(A,2));
P2 = MakeNormal(P2);
 P1= findLeft(A,P2,sigma,size(A,1),size(A,2));
n = input("top amount")
k = input("0 for random ")
A1=project(P1,sigma,P2,n,k);
n1 = findNorm(A1);
n2 = findNorm(A);
e = n2 - n1;
A=im2uint8(A);
A1 = im2uint8(A1);
error = A-A1;
error = im2uint8(error);
figure(1)
imshow(A1)
title("image with random 25% s.vector");
figure(2)
imshow(error)
title("error image with random 25% s.vector");
%x = graphplot(P1,sigma,P2,n2); 



