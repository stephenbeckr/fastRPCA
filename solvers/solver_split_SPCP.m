function [L,S,errHist] = solver_split_SPCP(X,params)
tic;

[m, n]   = size(X);
params.m = m;
params.n = n;

errFcn = setOpts(params,'errFcn',[]);
k      = setOpts(params,'k',10);
U0     = setOpts(params,'U0',randn(m,k));
V0     = setOpts(params,'V0',randn(n,k));
gpu    = setOpts(params,'gpu',0);

if ~isfield(params, 'lambdaS')
    params.lambdaS = 0.8;
elseif isempty(params.lambdaS)
    params.lambdaS = 0.8;
end

if ~isfield(params, 'lambdaL')
    params.lambdaL = 115;
elseif isempty(params.lambdaL)
    params.lambdaL = 115;
end

if strcmp(class(X), 'gpuArray')
	gpu = 1;
end

params.gpu = gpu;

if gpu
    U0 = gpuArray(U0);
    V0 = gpuArray(V0);
end

R = [vec(U0); vec(V0)];

ObjFunc = @(x)func_split_spcp(x,X,params,errFcn);

func_split_spcp();

[x,~,~] = lbfgs_gpu(ObjFunc,R,params);

errHist=func_split_spcp();
if ~isempty( errHist )
    figure;
    semilogy( errHist );
end

U = reshape(x(1:m*k),m,k);
V = reshape(x(m*k+1:m*k+k*n),n,k);
S = func_split_spcp(x,X,params,'S');
S = reshape(S,m,n);

L = U*V';

end



function out = setOpts(options, opt, default)
    if isfield(options, opt)
        out = options.(opt);
    else
        out = default;
    end
end