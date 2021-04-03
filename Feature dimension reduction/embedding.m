function y= embedding(D)
D=D'
[~,m]=size(D)
T2=corrcoef(D)-eye(m)
AJ1=ones(m)-eye(m)
H1=AANE_fun(AJ1,T2,1)
y=D*H1
end



