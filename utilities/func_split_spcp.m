function [f, df] = func_split_spcp(x,Y,params,errFunc)
% [f, df] = BurMontRPCA(x,Y,params,errFunc)
% [errHist] = BurMontRPCA();
% [S] = BurMontRPCA(x,Y,'S');

persistent errHist

if nargin==0
   f = errHist;
   errHist = [];
   return;
end
if nargin<3, params=[]; end
m   = params.m;
n   = params.n;
k   = params.k;

function out = setOpts( field, default )
    if ~isfield( params, field )
        params.(field)    = default;
    end
    out = params.(field);
    params    = rmfield( params, field ); % so we can do a check later
end

lambdaL = params.lambdaL;
lambdaS = params.lambdaS;

U = reshape(x(1:m*k),m,k);
V = reshape(x(m*k+1:m*k+n*k),n,k);

L   = U*V';
LY = vec(Y-L);

soft_thresh  = @(LY,lambdaS) sign(LY).*max(abs(LY) - lambdaS,0);
S = soft_thresh(LY,lambdaS);
if nargin>=3 && strcmpi(errFunc,'s')
    f = S;
    return;
end
SLY = reshape(S-LY,m,n);

fS = norm(SLY,'fro')^2/2 + lambdaS*norm(S(:),1);

f = lambdaL/2*(norm(U,'fro')^2 + norm(V,'fro')^2) + fS;

if nargout > 1
  df = gpuArray.zeros(m*k + n*k,1);
  df(1:m*k) = vec(lambdaL*U) + vec((SLY)*V);
  df(m*k+1:m*k+n*k) = vec(lambdaL*V) + vec((SLY)'*U);
end

if nargin >= 4 && ~isempty(errFunc)
    errHist(end+1,1) = toc;
    errHist(end,2) = gather(f);
clear U V

tic;
end

end
