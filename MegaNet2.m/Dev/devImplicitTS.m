clear
close all

A = randn(5,3);
h = 0.1;
I = eye(3);
Y = randn(3,6);

% the time stepping implicit function
f = @(A,Y) (h*A'*A + I)\Y; 
Z = f(A,Y);

T = @(A,h)(h*A'*A + I);
df = @(A,Y,V) -1/h * (T(A,h)\ ((V'*A + A'*V)*(T(A,h)\Y)));

%df1 = @(A,Y,V) -1/h * (T(A,h)\ ((A'*V)*(T(A,h)\Y)));
%df2 = @(A,Y,V) -1/h * (T(A,h)\ ((V'*A)*(T(A,h)\Y)));

dTf = @(A,Y,W) -1/h * A*( (T(A,h)\W)*((T(A,h)\Y)') + (T(A,h)\Y)*((T(A,h)\W)')); 
%dTf1 = @(A,Y,W) -1/h * (A*(T(A,h)\W) * ((T(A,h)\Y)'));
%dTf2 = @(A,Y,W) -1/h * (A*(T(A,h)\(Y*W'))*inv(T(A,h)));
%dTf2 = @(A,Y,W) -1/h * (A*(T(A,h)\Y)*(T(A,h)\W)');



% test
V = randn(size(A))*1e-3;
norm(f(A+V,Y)-f(A,Y),'fro')
norm(f(A+V,Y)-f(A,Y) - df(A,Y,V),'fro')

W = randn(size(Y));
%[vec(df1(A,Y,V))'*vec(W), vec(dTf1(A,Y,W))'*vec(V)]
%[vec(df2(A,Y,V))'*vec(W), vec(dTf2(A,Y,W))'*vec(V)]
[vec(df(A,Y,V))'*vec(W), vec(dTf(A,Y,W))'*vec(V)]
%[vec(df2(A,Y,V))'*vec(W), vec(dTf2(A,Y,W))'*vec(V)]