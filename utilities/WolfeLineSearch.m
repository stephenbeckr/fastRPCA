function [t,f_new,g_new,funEvals] = WolfeLineSearch(...
    x,t,d,f,g,gtd,c1,c2,maxLS,progTol,funObj)
%
% Bracketing Line Search to Satisfy Wolfe Conditions
%
% Based on code from ''minFunc'' (Mark Schmidt, 2005)
%
% Inputs:
%   x: starting location
%   t: initial step size
%   d: descent direction
%   f: function value at starting location
%   g: gradient at starting location
%   gtd: directional derivative at starting location
%   c1: sufficient decrease parameter
%   c2: curvature parameter
%   debug: display debugging information
%   LS_interp: type of interpolation
%   maxLS: maximum number of iterations
%   progTol: minimum allowable step length
%   doPlot: do a graphical display of interpolation
%   funObj: objective function
%   varargin: parameters of objective function
%
% Outputs:
%   t: step length
%   f_new: function value at x+t*d
%   g_new: gradient value at x+t*d
%   funEvals: number function evaluations performed by line search
%   H: Hessian at initial guess (only computed if requested)

% Evaluate the Objective and Gradient at the Initial Step
[f_new,g_new] = funObj(x+t*d);

funEvals = 1;
gtd_new = g_new'*d;

% Bracket an Interval containing a point satisfying the
% Wolfe criteria

LSiter = 0;
t_prev = 0;
f_prev = f;
g_prev = g;
nrmD = max(abs(d));
done = 0;

while LSiter < maxLS

    %% Bracketing Phase

    if f_new > f + c1*t*gtd || (LSiter > 1 && f_new >= f_prev)
        bracket = [t_prev t];
        bracketFval = [f_prev f_new];
        bracketGval = [g_prev g_new];
        break;
    elseif abs(gtd_new) <= -c2*gtd
        bracket = t;
        bracketFval = f_new;
        bracketGval = g_new;
        done = 1;
        break;
    elseif gtd_new >= 0
        bracket = [t_prev t];
        bracketFval = [f_prev f_new];
        bracketGval = [g_prev g_new];
        break;
    end
    t_prev = t;
    maxStep = t*10;

    t = maxStep;
    
    f_prev = f_new;
    g_prev = g_new;
    
    [f_new,g_new] = funObj(x + t*d);
    
    funEvals = funEvals + 1;
    gtd_new = g_new'*d;
    LSiter = LSiter+1;
end

if LSiter == maxLS
    bracket = [0 t];
    bracketFval = [f f_new];
    bracketGval = [g g_new];
end

%% Zoom Phase

% We now either have a point satisfying the criteria, or a bracket
% surrounding a point satisfying the criteria
% Refine the bracket until we find a point satisfying the criteria
insufProgress = 0;

while ~done && LSiter < maxLS

    % Find High and Low Points in bracket
    [f_LO, LOpos] = min(bracketFval);
    HIpos = -LOpos + 3;

    % Compute new trial value
    t = mean(bracket);


    % Test that we are making sufficient progress
    if min(max(bracket)-t,t-min(bracket))/(max(bracket)-min(bracket)) < 0.1
        if insufProgress || t>=max(bracket) || t <= min(bracket)
            if debug
                fprintf(', Evaluating at 0.1 away from boundary\n');
            end
            if abs(t-max(bracket)) < abs(t-min(bracket))
                t = max(bracket)-0.1*(max(bracket)-min(bracket));
            else
                t = min(bracket)+0.1*(max(bracket)-min(bracket));
            end
            insufProgress = 0;
        else
            insufProgress = 1;
        end
    else
        insufProgress = 0;
    end

    % Evaluate new point
    [f_new,g_new] = funObj(x + t*d);

    funEvals = funEvals + 1;
    gtd_new = g_new'*d;
    LSiter = LSiter+1;

	armijo = f_new < f + c1*t*gtd;
    if ~armijo || f_new >= f_LO
        % Armijo condition not satisfied or not lower than lowest
        % point
        bracket(HIpos) = t;
        bracketFval(HIpos) = f_new;
        bracketGval(:,HIpos) = g_new;
    else
        if abs(gtd_new) <= - c2*gtd
            % Wolfe conditions satisfied
            done = 1;
        elseif gtd_new*(bracket(HIpos)-bracket(LOpos)) >= 0
            % Old HI becomes new LO
            bracket(HIpos) = bracket(LOpos);
            bracketFval(HIpos) = bracketFval(LOpos);
            bracketGval(:,HIpos) = bracketGval(:,LOpos);
        end
        % New point becomes new LO
        bracket(LOpos) = t;
        bracketFval(LOpos) = f_new;
        bracketGval(:,LOpos) = g_new;
	end

    if ~done && abs(bracket(1)-bracket(2))*nrmD < progTol
        break;
    end

end

%%

[~, LOpos] = min(bracketFval);
t = bracket(LOpos);
f_new = bracketFval(LOpos);
g_new = bracketGval(:,LOpos);

end