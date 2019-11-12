function [f, df] = func_split_spcp(x,Y,params,errFcn)
% [f, df] = func_split_spcp(x,Y,params,errFunc)
% [errHist] = func_split_spcp();
% [S] = func_split_spcp(x,Y,params,'S');
%
% Compute function and gradient of split-SPCP objective
%
%   lambda_L/2 (||U||_F^2 + ||V||_F^2) + phi(U,V)
%
% where
%
%   phi(U,V) = min_S .5|| U*V' + S - Y ||_F^2 + lambda_S ||S||_1.

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
lambdaL = params.lambdaL;
lambdaS = params.lambdaS;
useGPU  = params.gpu;

U = reshape(x(1:m*k),m,k);
V = reshape(x(m*k+1:m*k+n*k),n,k);

L   = U*V';
LY = vec(Y-L);

soft_thresh  = @(LY,lambdaS) sign(LY).*max(abs(LY) - lambdaS,0);
S = soft_thresh(LY,lambdaS);
if nargout==1
    f = S;
    return;
end
SLY = reshape(S-LY,m,n);

fS = norm(SLY,'fro')^2/2 + lambdaS*norm(S(:),1);

f = lambdaL/2*(norm(U,'fro')^2 + norm(V,'fro')^2) + fS;

if nargout > 1
    
    if useGPU
        df = gpuArray.zeros(m*k + n*k,1);
    else
        df = zeros(m*k + n*k,1);
    end
    
    df(1:m*k) = vec(lambdaL*U) + vec((SLY)*V);
    df(m*k+1:m*k+n*k) = vec(lambdaL*V) + vec((SLY)'*U);

end

errHist(end+1,1) = toc;
errHist(end,2) = gather(f);

if ~isempty(errFcn)
    errHist(end,3) = errFcn(x);
end

tic;

end



function out = setOpts( params, field, default )
    if ~isfield( params, field )
        params.(field)    = default;
    end
    out = params.(field);
    params    = rmfield( params, field ); % so we can do a check later
end
