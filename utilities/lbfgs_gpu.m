function [x,f,exitflag] = lbfgs_gpu(funObj,x0,options)
% [x,f,exitflag] = lbfgs_gpu(funObj,x0,options)
%
% Limited-memory BFGS with Wolfe line-search and GPU support.
% Based on code from ''minFunc'' (Mark Schmidt, 2005)
%
% Options:
%   GPU     - 0 is off-GPU, 1 is on-GPU
%   maxIter - number of iterations
%   store   - number of corrections to store in memory
%             (more store lead to faster convergence).
%   verbose - 0 turns off printing, 1 turns on printing
%   c1      - Sufficient Decrease for Armijo condition (1e-4)
%   c2      - Curvature Decrease for Wolfe conditions (0.9)
%   progTol - Termination tolerance on gradient size (1e-8)
%   optTol  - Termination tolerance on objective decrease (1e-9)
%
% Inputs:
%   funObj - is a function handle (objective/gradient map)
%   x0 - is a starting vector;
%   options - is a struct containing parameters
%
% Outputs:
%   x is the minimum value found
%   f is the function value at the minimum found
%   exitflag returns an exit condition
%

if nargin < 3
    options = [];
end

% Set parameters
maxIter = setOpts(options,'MaxIter',100);
verbose = setOpts(options,'verbose',1);
store   = setOpts(options,'store',100);
gpu     = setOpts(options,'gpu',0);
c1      = setOpts(options,'c1',1e-4);
c2      = setOpts(options,'c2',0.9);
progTol = setOpts(options,'progTol',1e-9);
optTol  = setOpts(options,'optTol',1e-8);

% Initialize
p = length(x0);
d = zeros(p,1);
x = x0;
t = 1;

funEvalMultiplier = 1;

% Evaluate Initial Point
[f,g] = funObj(x);
funEvals = 1;

% Output Log
if verbose
    fprintf('%10s %10s %15s %15s %15s\n','Iteration','FunEvals','Step Length','Function Val','Opt Cond');
end

% Compute optimality of initial point
optCond = max(abs(g));

% Exit if initial point is optimal
if optCond <= optTol
    exitflag=1;
    msg = 'Optimality Condition below optTol';
    if verbose
        fprintf('%s\n',msg);
    end
    return;
end


% Perform up to a maximum of 'maxIter' descent steps:
for i = 1:maxIter

    % ****************** COMPUTE DESCENT DIRECTION *****************
    if i == 1
        d = -g; % Initially use steepest descent direction
        
        if gpu
            S = gpuArray.zeros(p,store);
            Y = gpuArray.zeros(p,store);
            YS = gpuArray.zeros(store,1);
        else
            S = zeros(p,store);
            Y = zeros(p,store);
            YS = zeros(store,1);
        end
        
        lbfgs_start = 1;
        lbfgs_end = 0;
        Hdiag = 1;
    else
        [S,Y,YS,lbfgs_start,lbfgs_end,Hdiag,skipped] = lbfgsAdd(g-g_old,t*d,S,Y,YS,lbfgs_start,lbfgs_end,Hdiag);
        d = lbfgsProd(g,S,Y,YS,lbfgs_start,lbfgs_end,Hdiag);
    end
    g_old = g;

    % ****************** COMPUTE STEP LENGTH ************************

    % Directional Derivative
    gtd = g'*d;

    % Check that progress can be made along direction
    if gtd > -progTol
        exitflag=2;
        msg = 'Directional Derivative below progTol';
        break;
    end

    % Select Initial Guess for Line Search
    if i == 1
        t = min(1,1/sum(abs(g)));
    else
        t = 1;
    end

    % Line Search
    f_old = f;
    
    % Find Point satisfying Wolfe conditions
    [t,f,g,LSfunEvals] = WolfeLineSearch(x,t,d,f,g,gtd,c1,c2,25,progTol,funObj);
    funEvals = funEvals + LSfunEvals;
    x = x + t*d;

	% Compute Optimality Condition
	optCond = max(abs(g));
	
    % Output iteration information
    if verbose
        fprintf('%10d %10d %15.5e %15.5e %15.5e\n',i,funEvals*funEvalMultiplier,t,f,optCond);
    end
	
    % Check Optimality Condition
    if optCond <= optTol
        exitflag=1;
        msg = 'Optimality Condition below optTol';
        break;
    end

    % ******************* Check for lack of progress *******************

    if max(abs(t*d)) <= progTol
        exitflag=2;
        msg = 'Step Size below progTol';
        break;
    end


    if abs(f-f_old) < progTol
        exitflag=2;
        msg = 'Function Value changing by less than progTol';
        break;
    end

    % ******** Check for going over iteration/evaluation limit *******************
    if i == maxIter
        exitflag = 0;
        msg='Reached Maximum Number of Iterations';
        break;
    end

end

if verbose
    fprintf('%s\n',msg);
end

end





function out = setOpts(options, opt, default)
    if isfield(options, opt)
        out = options.(opt);
    else
        out = default;
    end
end

